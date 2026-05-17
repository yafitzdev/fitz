# fitz_sage/engines/fitz_krag/ingestion/strategies/typescript.py
"""
TypeScript/JavaScript ingestion strategy using tree-sitter.

Extracts functions, classes, interfaces, type aliases, and import relationships.
Requires: tree-sitter, tree-sitter-typescript (optional dependency).
Falls back to regex-based extraction when tree-sitter is not installed.
"""

from __future__ import annotations

import logging
import re

from fitz_sage.engines.fitz_krag.ingestion.strategies.base import (
    ImportEdge,
    IngestResult,
    SymbolEntry,
)

logger = logging.getLogger(__name__)

try:
    import tree_sitter_typescript as ts_typescript
    from tree_sitter import Language, Parser

    TS_LANGUAGE = Language(ts_typescript.language_typescript())
    TSX_LANGUAGE = Language(ts_typescript.language_tsx())
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False

_warned_no_tree_sitter = False

# --- Regex patterns for fallback ---
_FUNC_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
    re.MULTILINE,
)
_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_INTERFACE_RE = re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)", re.MULTILINE)
_TYPE_RE = re.compile(r"^\s*(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE)
_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>",
    re.MULTILINE,
)
_IMPORT_RE = re.compile(
    r"""^\s*import\s+(?:.*?\s+from\s+)?['"]([^'"]+)['"]""",
    re.MULTILINE,
)


class TypeScriptIngestStrategy:
    """Extracts symbols from TypeScript/JavaScript using tree-sitter."""

    def content_types(self) -> set[str]:
        return {".ts", ".tsx", ".js", ".jsx"}

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from TypeScript/JavaScript source."""
        if not _HAS_TREE_SITTER:
            return _regex_fallback(source, file_path)

        try:
            lang = TSX_LANGUAGE if file_path.endswith((".tsx", ".jsx")) else TS_LANGUAGE
            parser = Parser(lang)
            tree = parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning(f"Parse error for {file_path}: {e}")
            return _regex_fallback(source, file_path)

        lines = source.splitlines()
        module_name = _path_to_module(file_path)
        symbols: list[SymbolEntry] = []
        imports: list[ImportEdge] = []

        _walk_node(tree.root_node, lines, module_name, symbols, imports)
        return IngestResult(symbols=symbols, imports=imports)


def _regex_fallback(source: str, file_path: str) -> IngestResult:
    """Extract symbols via regex when tree-sitter is unavailable."""
    global _warned_no_tree_sitter
    if not _warned_no_tree_sitter:
        logger.warning(
            "tree-sitter-typescript not installed, using regex fallback. "
            "Install with: pip install fitz-sage[krag-typescript]"
        )
        _warned_no_tree_sitter = True

    module_name = _path_to_module(file_path)
    symbols: list[SymbolEntry] = []
    imports: list[ImportEdge] = []

    for m in _FUNC_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="function",
                start_line=line_no,
                end_line=line_no,
                signature=f"function {m.group(1)}()",
            )
        )

    for m in _CLASS_RE.finditer(source):
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

    for m in _INTERFACE_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="interface",
                start_line=line_no,
                end_line=line_no,
                signature=f"interface {m.group(1)}",
            )
        )

    for m in _TYPE_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="type",
                start_line=line_no,
                end_line=line_no,
                signature=f"type {m.group(1)}",
            )
        )

    for m in _ARROW_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="function",
                start_line=line_no,
                end_line=line_no,
                signature=f"const {m.group(1)} = () => ...",
            )
        )

    for m in _IMPORT_RE.finditer(source):
        imports.append(ImportEdge(target_module=m.group(1), import_names=[]))

    return IngestResult(symbols=symbols, imports=imports)


# --- Tree-sitter extraction (used when tree-sitter is available) ---


def _walk_node(node, lines, module_name, symbols, imports):
    """Recursively walk tree-sitter nodes to extract symbols."""
    for child in node.children:
        node_type = child.type

        if node_type == "function_declaration":
            sym = _extract_function(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "class_declaration":
            sym = _extract_class(child, lines, module_name)
            if sym:
                symbols.append(sym)
                body = child.child_by_field_name("body")
                if body:
                    _walk_class_body(body, lines, module_name, sym.name, symbols)

        elif node_type == "interface_declaration":
            sym = _extract_interface(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "type_alias_declaration":
            sym = _extract_type_alias(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "lexical_declaration":
            extracted = _extract_lexical(child, lines, module_name)
            symbols.extend(extracted)

        elif node_type == "export_statement":
            _walk_node(child, lines, module_name, symbols, imports)

        elif node_type == "import_statement":
            edge = _extract_import(child)
            if edge:
                imports.append(edge)


def _walk_class_body(body_node, lines, module_name, class_name, symbols):
    """Extract methods from a class body."""
    for child in body_node.children:
        if child.type in ("method_definition", "public_field_definition"):
            sym = _extract_method(child, lines, module_name, class_name)
            if sym:
                symbols.append(sym)


def _extract_function(node, lines, module_name):
    """Extract a function declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    params = node.child_by_field_name("parameters")
    sig = f"function {name}{_node_text(params) if params else '()'}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="function",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_class(node, lines, module_name):
    """Extract a class declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    heritage = _get_heritage(node)
    sig = f"class {name}{heritage}" if heritage else f"class {name}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="class",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_interface(node, lines, module_name):
    """Extract an interface declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="interface",
        start_line=start,
        end_line=end,
        signature=f"interface {name}",
    )


def _extract_type_alias(node, lines, module_name):
    """Extract a type alias declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="type",
        start_line=start,
        end_line=end,
        signature=f"type {name}",
    )


def _extract_method(node, lines, module_name, class_name):
    """Extract a method from a class."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    params = node.child_by_field_name("parameters")
    sig = f"{name}{_node_text(params) if params else '()'}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{class_name}.{name}",
        kind="method",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_lexical(node, lines, module_name):
    """Extract symbols from const/let declarations."""
    symbols = []
    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if not name_node:
                continue

            name = _node_text(name_node)
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1

            if value_node and value_node.type == "arrow_function":
                params = value_node.child_by_field_name("parameters")
                sig = f"const {name} = {_node_text(params) if params else '()'} => ..."
                kind = "function"
            else:
                sig = f"const {name}"
                kind = "constant"

            symbols.append(
                SymbolEntry(
                    name=name,
                    qualified_name=f"{module_name}.{name}",
                    kind=kind,
                    start_line=start,
                    end_line=end,
                    signature=sig,
                )
            )
    return symbols


def _extract_import(node):
    """Extract import edge from an import statement."""
    source_node = node.child_by_field_name("source")
    if not source_node:
        return None

    module_path = _node_text(source_node).strip("'\"")
    import_names = []

    for child in node.children:
        if child.type == "import_clause":
            for sub in child.children:
                if sub.type == "identifier":
                    import_names.append(_node_text(sub))
                elif sub.type == "named_imports":
                    for spec in sub.children:
                        if spec.type == "import_specifier":
                            name_node = spec.child_by_field_name("name")
                            if name_node:
                                import_names.append(_node_text(name_node))

    return ImportEdge(target_module=module_path, import_names=import_names)


def _get_heritage(node):
    """Get heritage clause (extends/implements) text."""
    for child in node.children:
        if child.type == "class_heritage":
            return f" {_node_text(child)}"
    return ""


def _node_text(node) -> str:
    """Get text content of a tree-sitter node."""
    if node is None:
        return ""
    if isinstance(node.text, bytes):
        return node.text.decode("utf-8")
    return str(node.text)


def _path_to_module(file_path: str) -> str:
    """Convert file path to module-like name."""
    path = file_path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    for ext in (".ts", ".tsx", ".js", ".jsx"):
        if path.endswith(ext):
            path = path[: -len(ext)]
            break
    if path.endswith("/index"):
        path = path[: -len("/index")]
    return path.replace("/", ".")
