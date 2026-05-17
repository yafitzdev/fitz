# fitz_sage/engines/fitz_krag/ingestion/strategies/python_code.py
"""
Python code ingestion strategy using stdlib ast module.

Extracts functions, classes, methods, constants, and import relationships.
Zero external dependencies.
"""

from __future__ import annotations

import ast
import logging
import re

from fitz_sage.engines.fitz_krag.ingestion.strategies.base import (
    ImportEdge,
    IngestResult,
    SymbolEntry,
)

logger = logging.getLogger(__name__)


class PythonCodeIngestStrategy:
    """Extracts symbols from Python source using stdlib ast module."""

    def content_types(self) -> set[str]:
        return {".py"}

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from Python source."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}, using regex fallback")
            return _regex_fallback(source, file_path)

        lines = source.splitlines()
        module_name = _path_to_module(file_path)
        symbols: list[SymbolEntry] = []
        imports: list[ImportEdge] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(_extract_function(node, lines, module_name))

            elif isinstance(node, ast.ClassDef):
                # Class itself
                symbols.append(_extract_class(node, lines, module_name))
                # Methods and class-level constants
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(_extract_method(item, lines, module_name, node.name))
                    elif isinstance(item, ast.Assign):
                        entry = _extract_constant(item, lines, module_name, class_name=node.name)
                        if entry:
                            symbols.append(entry)

            elif isinstance(node, ast.Assign):
                entry = _extract_constant(node, lines, module_name)
                if entry:
                    symbols.append(entry)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportEdge(
                            target_module=alias.name,
                            import_names=[alias.asname or alias.name],
                        )
                    )

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # Relative import: resolve to absolute module name
                    resolved = _resolve_relative_import(
                        module_name, node.module, node.level, file_path
                    )
                    if resolved:
                        names = [alias.name for alias in node.names]
                        imports.append(ImportEdge(target_module=resolved, import_names=names))
                elif node.module:
                    names = [alias.name for alias in node.names]
                    imports.append(ImportEdge(target_module=node.module, import_names=names))

        return IngestResult(symbols=symbols, imports=imports)


# --- Regex fallback for files with syntax errors ---

_RX_FUNC = re.compile(r"^(?:async\s+)?def\s+(\w+)\s*\(", re.MULTILINE)
_RX_CLASS = re.compile(r"^class\s+(\w+)\s*[:\(]", re.MULTILINE)
_RX_CONST = re.compile(r"^([A-Z][A-Z0-9_]{2,})\s*=", re.MULTILINE)
_RX_IMPORT = re.compile(r"^(?:from\s+([\w.]+)\s+)?import\s+(\w+)", re.MULTILINE)


def _regex_fallback(source: str, file_path: str) -> IngestResult:
    """Extract symbols via regex when ast.parse() fails due to syntax errors."""
    module_name = _path_to_module(file_path)
    symbols: list[SymbolEntry] = []
    imports: list[ImportEdge] = []

    for m in _RX_FUNC.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="function",
                start_line=line_no,
                end_line=line_no,
                signature=f"def {m.group(1)}()",
            )
        )

    for m in _RX_CLASS.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="class",
                start_line=line_no,
                end_line=line_no,
                signature=f"class {m.group(1)}",
            )
        )

    for m in _RX_CONST.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="constant",
                start_line=line_no,
                end_line=line_no,
            )
        )

    for m in _RX_IMPORT.finditer(source):
        if m.group(1):  # "from X import Y"
            imports.append(ImportEdge(target_module=m.group(1), import_names=[m.group(2)]))
        else:  # "import X"
            imports.append(ImportEdge(target_module=m.group(2), import_names=[]))

    return IngestResult(symbols=symbols, imports=imports)


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    module_name: str,
) -> SymbolEntry:
    """Extract a top-level function."""
    start = node.lineno
    end = node.end_lineno or start
    sig = _build_signature(node)
    refs = _extract_references(node)

    return SymbolEntry(
        name=node.name,
        qualified_name=f"{module_name}.{node.name}",
        kind="function",
        start_line=start,
        end_line=end,
        signature=sig,
        references=refs,
    )


def _extract_class(
    node: ast.ClassDef,
    lines: list[str],
    module_name: str,
) -> SymbolEntry:
    """Extract a class (header + docstring, not full body)."""
    start = node.lineno
    end = node.end_lineno or start
    bases = [_name_of(b) for b in node.bases if _name_of(b)]
    sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
    refs = [_name_of(b) for b in node.bases if _name_of(b)]

    return SymbolEntry(
        name=node.name,
        qualified_name=f"{module_name}.{node.name}",
        kind="class",
        start_line=start,
        end_line=end,
        signature=sig,
        references=refs,
    )


def _extract_method(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    module_name: str,
    class_name: str,
) -> SymbolEntry:
    """Extract a method within a class."""
    start = node.lineno
    end = node.end_lineno or start
    sig = _build_signature(node)
    refs = _extract_references(node)

    return SymbolEntry(
        name=node.name,
        qualified_name=f"{module_name}.{class_name}.{node.name}",
        kind="method",
        start_line=start,
        end_line=end,
        signature=sig,
        references=refs,
    )


def _extract_constant(
    node: ast.Assign,
    lines: list[str],
    module_name: str,
    class_name: str | None = None,
) -> SymbolEntry | None:
    """Extract a module-level or class-level UPPER_CASE constant."""
    if len(node.targets) != 1:
        return None
    target = node.targets[0]
    if not isinstance(target, ast.Name):
        return None
    name = target.id
    if not re.match(r"^[A-Z][A-Z0-9_]*$", name):
        return None

    start = node.lineno
    end = node.end_lineno or start

    qualified = f"{module_name}.{class_name}.{name}" if class_name else f"{module_name}.{name}"

    return SymbolEntry(
        name=name,
        qualified_name=qualified,
        kind="constant",
        start_line=start,
        end_line=end,
        signature=None,
    )


def _build_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a function/method signature string from AST arguments."""
    args = node.args
    parts: list[str] = []

    # Positional args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        annotation = _annotation_str(arg.annotation)
        name = arg.arg
        if annotation:
            name = f"{name}: {annotation}"
        default_idx = i - defaults_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            name = f"{name}=..."
        parts.append(name)

    # *args
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")

    # **kwargs
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    ret = _annotation_str(node.returns)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    sig = f"{prefix} {node.name}({', '.join(parts)})"
    if ret:
        sig += f" -> {ret}"
    return sig


def _annotation_str(node: ast.expr | None) -> str:
    """Convert an annotation AST node to a readable string."""
    if node is None:
        return ""
    return ast.unparse(node)


def _name_of(node: ast.expr) -> str:
    """Get name from a Name or Attribute node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return ast.unparse(node)
    return ""


def _extract_references(node: ast.AST) -> list[str]:
    """Extract referenced names from a function/method body."""
    refs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            refs.add(child.id)
        elif isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name):
                refs.add(f"{child.value.id}.{child.attr}")
    return sorted(refs)


def _resolve_relative_import(
    module_name: str, import_module: str | None, level: int, file_path: str
) -> str | None:
    """Resolve a relative import to an absolute module name.

    ``from .models import User`` in file ``src/utils/helpers.py`` (module
    ``src.utils.helpers``) with level=1 resolves to ``src.utils.models``.

    For ``__init__.py`` files the module name already IS the package, so
    level=1 stays at the same package rather than going up one extra level.
    """
    parts = module_name.split(".")

    # For non-__init__ files, the package is the parent (drop last component).
    # For __init__.py the module name is already the package.
    is_init = file_path.replace("\\", "/").endswith("/__init__.py") or file_path == "__init__.py"
    if not is_init and parts:
        parts = parts[:-1]

    # level=1 means "current package", each extra level goes up one more
    up = level - 1
    if up > len(parts):
        return None  # can't go above root
    if up > 0:
        parts = parts[:-up]

    if import_module:
        parts.append(import_module)

    return ".".join(parts) if parts else None


def _path_to_module(file_path: str) -> str:
    """Convert file path to a Python module-like name."""
    # Strip leading ./ and trailing .py, replace / with .
    path = file_path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if path.endswith(".py"):
        path = path[:-3]
    if path.endswith("/__init__"):
        path = path[: -len("/__init__")]
    return path.replace("/", ".")
