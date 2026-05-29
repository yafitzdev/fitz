# fitz_sage/engines/fitz_krag/ingestion/strategies/go.py
"""
Go ingestion strategy using tree-sitter.

Extracts functions, methods (with receiver), types (struct, interface),
const/var declarations, and import relationships.
Requires: tree-sitter, tree-sitter-go (optional dependency).
Falls back to regex-based extraction when tree-sitter is not installed.
"""

from __future__ import annotations

import logging
import re

from fitz_sage.engines.fitz_krag.ingestion.code_utils import node_text, path_to_module
from fitz_sage.engines.fitz_krag.ingestion.strategies.base import (
    ImportEdge,
    IngestResult,
    SymbolEntry,
)

logger = logging.getLogger(__name__)

try:
    import tree_sitter_go as ts_go
    from tree_sitter import Language, Parser

    GO_LANGUAGE = Language(ts_go.language())
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False

_warned_no_tree_sitter = False

# --- Regex patterns for fallback ---
_FUNC_RE = re.compile(r"^func\s+(\w+)\s*\(", re.MULTILINE)
_METHOD_RE = re.compile(r"^func\s+\([^)]+\)\s+(\w+)\s*\(", re.MULTILINE)
_STRUCT_RE = re.compile(r"^type\s+(\w+)\s+struct\b", re.MULTILINE)
_IFACE_RE = re.compile(r"^type\s+(\w+)\s+interface\b", re.MULTILINE)
_TYPE_RE = re.compile(r"^type\s+(\w+)\s+\w", re.MULTILINE)
_CONST_RE = re.compile(r"^\s*(\w+)\s*(?:=|[A-Z])", re.MULTILINE)
_PACKAGE_RE = re.compile(r"^package\s+(\w+)", re.MULTILINE)
_IMPORT_RE = re.compile(r'"([^"]+)"', re.MULTILINE)


class GoIngestStrategy:
    """Extracts symbols from Go source using tree-sitter."""

    def content_types(self) -> set[str]:
        return {".go"}

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from Go source."""
        if not _HAS_TREE_SITTER:
            return _regex_fallback(source, file_path)

        try:
            parser = Parser(GO_LANGUAGE)
            tree = parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning(f"Parse error for {file_path}: {e}")
            return _regex_fallback(source, file_path)

        lines = source.splitlines()
        package = _extract_package(tree.root_node)
        module_name = package or path_to_module(file_path, (".go",))
        symbols: list[SymbolEntry] = []
        imports: list[ImportEdge] = []

        for child in tree.root_node.children:
            node_type = child.type

            if node_type == "function_declaration":
                sym = _extract_function(child, lines, module_name)
                if sym:
                    symbols.append(sym)

            elif node_type == "method_declaration":
                sym = _extract_method(child, lines, module_name)
                if sym:
                    symbols.append(sym)

            elif node_type == "type_declaration":
                extracted = _extract_type_decl(child, lines, module_name)
                symbols.extend(extracted)

            elif node_type == "const_declaration":
                extracted = _extract_const_var(child, lines, module_name, "constant")
                symbols.extend(extracted)

            elif node_type == "var_declaration":
                extracted = _extract_const_var(child, lines, module_name, "variable")
                symbols.extend(extracted)

            elif node_type == "import_declaration":
                edges = _extract_imports(child)
                imports.extend(edges)

        return IngestResult(symbols=symbols, imports=imports)


def _regex_extract_block(lines: list[str], start_line: int) -> tuple[str, int]:
    """Extract a brace-delimited block starting from a given line.

    Returns (source_text, end_line) where end_line is 1-indexed.
    Walks forward from start_line counting braces until the opening brace
    is closed. Falls back to a 20-line window if no braces found.
    """
    depth = 0
    found_open = False
    end = start_line  # 1-indexed

    for i in range(start_line - 1, min(len(lines), start_line + 50)):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
        end = i + 1
        if found_open and depth <= 0:
            break

    if not found_open:
        end = min(len(lines), start_line + 20)

    return "\n".join(lines[start_line - 1 : end]), end


def _regex_fallback(source: str, file_path: str) -> IngestResult:
    """Extract symbols via regex when tree-sitter is unavailable."""
    global _warned_no_tree_sitter
    if not _warned_no_tree_sitter:
        logger.warning(
            "tree-sitter-go not installed, using regex fallback. "
            "Install with: pip install fitz-sage[krag-go]"
        )
        _warned_no_tree_sitter = True

    pkg_match = _PACKAGE_RE.search(source)
    module_name = pkg_match.group(1) if pkg_match else path_to_module(file_path, (".go",))
    lines = source.splitlines()

    symbols: list[SymbolEntry] = []
    imports: list[ImportEdge] = []

    # Track structs/interfaces to exclude from generic type matches
    seen_type_names: set[str] = set()

    for m in _FUNC_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        _, end_line = _regex_extract_block(lines, line_no)
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="function",
                start_line=line_no,
                end_line=end_line,
                signature=f"func {m.group(1)}()",
            )
        )

    for m in _METHOD_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        _, end_line = _regex_extract_block(lines, line_no)
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="method",
                start_line=line_no,
                end_line=end_line,
                signature=f"func (...) {m.group(1)}()",
            )
        )

    for m in _STRUCT_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        _, end_line = _regex_extract_block(lines, line_no)
        seen_type_names.add(m.group(1))
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="struct",
                start_line=line_no,
                end_line=end_line,
                signature=f"type {m.group(1)} struct",
            )
        )

    for m in _IFACE_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        _, end_line = _regex_extract_block(lines, line_no)
        seen_type_names.add(m.group(1))
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="interface",
                start_line=line_no,
                end_line=end_line,
                signature=f"type {m.group(1)} interface",
            )
        )

    # Generic type aliases (exclude already-captured structs/interfaces)
    for m in _TYPE_RE.finditer(source):
        name = m.group(1)
        if name in seen_type_names:
            continue
        line_no = source[: m.start()].count("\n") + 1
        _, end_line = _regex_extract_block(lines, line_no)
        symbols.append(
            SymbolEntry(
                name=name,
                qualified_name=f"{module_name}.{name}",
                kind="type",
                start_line=line_no,
                end_line=end_line,
                signature=f"type {name}",
            )
        )

    # Extract import paths from import blocks
    import_block = re.search(r"^import\s*\((.*?)\)", source, re.MULTILINE | re.DOTALL)
    if import_block:
        for m in _IMPORT_RE.finditer(import_block.group(1)):
            path = m.group(1)
            name = path.rsplit("/", 1)[-1]
            imports.append(ImportEdge(target_module=path, import_names=[name]))
    # Single-line imports
    for m in re.finditer(r'^import\s+"([^"]+)"', source, re.MULTILINE):
        path = m.group(1)
        name = path.rsplit("/", 1)[-1]
        imports.append(ImportEdge(target_module=path, import_names=[name]))

    return IngestResult(symbols=symbols, imports=imports)


# --- Tree-sitter extraction (used when tree-sitter is available) ---


def _extract_function(node, lines, module_name):
    """Extract a top-level function declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    params = node.child_by_field_name("parameters")
    result = node.child_by_field_name("result")
    sig = f"func {name}{node_text(params) if params else '()'}"
    if result:
        sig += f" {node_text(result)}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="function",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_method(node, lines, module_name):
    """Extract a method declaration (function with receiver)."""
    name_node = node.child_by_field_name("name")
    receiver_node = node.child_by_field_name("receiver")
    if not name_node:
        return None

    name = node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    receiver_type = _extract_receiver_type(receiver_node)
    params = node.child_by_field_name("parameters")
    result = node.child_by_field_name("result")

    receiver_str = f"({node_text(receiver_node)})" if receiver_node else "()"
    sig = f"func {receiver_str} {name}{node_text(params) if params else '()'}"
    if result:
        sig += f" {node_text(result)}"

    qualified = (
        f"{module_name}.{receiver_type}.{name}" if receiver_type else f"{module_name}.{name}"
    )

    return SymbolEntry(
        name=name,
        qualified_name=qualified,
        kind="method",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_type_decl(node, lines, module_name):
    """Extract type declarations (struct, interface, etc.)."""
    symbols = []
    for child in node.children:
        if child.type == "type_spec":
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if not name_node:
                continue

            name = node_text(name_node)
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1

            if type_node and type_node.type == "struct_type":
                kind = "struct"
                sig = f"type {name} struct"
            elif type_node and type_node.type == "interface_type":
                kind = "interface"
                sig = f"type {name} interface"
            else:
                kind = "type"
                sig = f"type {name}"

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


def _extract_const_var(node, lines, module_name, kind):
    """Extract const or var declarations."""
    symbols = []
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    for child in node.children:
        if child.type in ("const_spec", "var_spec"):
            name_node = child.child_by_field_name("name")
            if name_node:
                name = node_text(name_node)
                symbols.append(
                    SymbolEntry(
                        name=name,
                        qualified_name=f"{module_name}.{name}",
                        kind=kind,
                        start_line=start,
                        end_line=end,
                        signature=f"{kind} {name}",
                    )
                )
    return symbols


def _extract_imports(node):
    """Extract import edges from an import declaration."""
    edges = []
    for child in node.children:
        if child.type == "import_spec":
            path_node = child.child_by_field_name("path")
            if path_node:
                path = node_text(path_node).strip('"')
                name = path.rsplit("/", 1)[-1]
                edges.append(ImportEdge(target_module=path, import_names=[name]))
        elif child.type == "import_spec_list":
            for spec in child.children:
                if spec.type == "import_spec":
                    path_node = spec.child_by_field_name("path")
                    if path_node:
                        path = node_text(path_node).strip('"')
                        name = path.rsplit("/", 1)[-1]
                        edges.append(ImportEdge(target_module=path, import_names=[name]))
    return edges


def _extract_package(root_node):
    """Extract package name from root node."""
    for child in root_node.children:
        if child.type == "package_clause":
            for sub in child.children:
                if sub.type == "package_identifier":
                    return node_text(sub)
    return None


def _extract_receiver_type(receiver_node):
    """Extract the type name from a method receiver."""
    if not receiver_node:
        return None
    text = node_text(receiver_node)
    text = text.strip("()")
    parts = text.split()
    for part in reversed(parts):
        clean = part.lstrip("*")
        if clean and clean[0].isupper():
            return clean
    return None
