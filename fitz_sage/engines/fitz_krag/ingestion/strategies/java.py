# fitz_sage/engines/fitz_krag/ingestion/strategies/java.py
"""
Java ingestion strategy using tree-sitter.

Extracts classes, interfaces, enums, records, methods, constructors, fields,
and import relationships.
Requires: tree-sitter, tree-sitter-java (optional dependency).
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
    import tree_sitter_java as ts_java
    from tree_sitter import Language, Parser

    JAVA_LANGUAGE = Language(ts_java.language())
    _HAS_TREE_SITTER = True
except ImportError:
    _HAS_TREE_SITTER = False

_warned_no_tree_sitter = False

# --- Regex patterns for fallback ---
_CLASS_RE = re.compile(
    r"^\s*(?:public|private|protected)?\s*(?:abstract\s+)?(?:final\s+)?class\s+(\w+)",
    re.MULTILINE,
)
_INTERFACE_RE = re.compile(r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)", re.MULTILINE)
_ENUM_RE = re.compile(r"^\s*(?:public|private|protected)?\s*enum\s+(\w+)", re.MULTILINE)
_RECORD_RE = re.compile(r"^\s*(?:public|private|protected)?\s*record\s+(\w+)", re.MULTILINE)
_METHOD_RE = re.compile(
    r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?"
    r"(?:abstract\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]+>)?(?:\[\])*)\s+(\w+)\s*\(",
    re.MULTILINE,
)
_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([a-zA-Z_][\w.]*(?:\.\*)?)\s*;", re.MULTILINE)
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)


class JavaIngestStrategy:
    """Extracts symbols from Java source using tree-sitter."""

    def content_types(self) -> set[str]:
        return {".java"}

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from Java source."""
        if not _HAS_TREE_SITTER:
            return _regex_fallback(source, file_path)

        try:
            parser = Parser(JAVA_LANGUAGE)
            tree = parser.parse(source.encode("utf-8"))
        except Exception as e:
            logger.warning(f"Parse error for {file_path}: {e}")
            return _regex_fallback(source, file_path)

        lines = source.splitlines()
        package = _extract_package(tree.root_node)
        module_name = package or _path_to_module(file_path)
        symbols: list[SymbolEntry] = []
        imports: list[ImportEdge] = []

        _walk_node(tree.root_node, lines, module_name, symbols, imports)
        return IngestResult(symbols=symbols, imports=imports)


def _regex_fallback(source: str, file_path: str) -> IngestResult:
    """Extract symbols via regex when tree-sitter is unavailable."""
    global _warned_no_tree_sitter
    if not _warned_no_tree_sitter:
        logger.warning(
            "tree-sitter-java not installed, using regex fallback. "
            "Install with: pip install fitz-sage[krag-java]"
        )
        _warned_no_tree_sitter = True

    # Detect package for qualified names
    pkg_match = _PACKAGE_RE.search(source)
    module_name = pkg_match.group(1) if pkg_match else _path_to_module(file_path)

    symbols: list[SymbolEntry] = []
    imports: list[ImportEdge] = []

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

    for m in _ENUM_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="enum",
                start_line=line_no,
                end_line=line_no,
                signature=f"enum {m.group(1)}",
            )
        )

    for m in _RECORD_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=m.group(1),
                qualified_name=f"{module_name}.{m.group(1)}",
                kind="record",
                start_line=line_no,
                end_line=line_no,
                signature=f"record {m.group(1)}",
            )
        )

    for m in _METHOD_RE.finditer(source):
        name = m.group(1)
        # Skip false positives: Java keywords that look like method return types
        if name in ("if", "else", "for", "while", "switch", "return", "new", "class"):
            continue
        line_no = source[: m.start()].count("\n") + 1
        symbols.append(
            SymbolEntry(
                name=name,
                qualified_name=f"{module_name}.{name}",
                kind="method",
                start_line=line_no,
                end_line=line_no,
                signature=f"{name}()",
            )
        )

    for m in _IMPORT_RE.finditer(source):
        full_import = m.group(1)
        parts = full_import.rsplit(".", 1)
        if len(parts) == 2:
            module, name = parts
            imports.append(
                ImportEdge(
                    target_module=module,
                    import_names=[name] if name != "*" else [],
                )
            )
        else:
            imports.append(ImportEdge(target_module=full_import, import_names=[]))

    return IngestResult(symbols=symbols, imports=imports)


# --- Tree-sitter extraction (used when tree-sitter is available) ---


def _walk_node(node, lines, module_name, symbols, imports, class_name=None):
    """Recursively walk tree-sitter nodes to extract symbols."""
    for child in node.children:
        node_type = child.type

        if node_type == "class_declaration":
            sym = _extract_class(child, lines, module_name)
            if sym:
                symbols.append(sym)
                body = child.child_by_field_name("body")
                if body:
                    _walk_node(body, lines, module_name, symbols, imports, sym.name)

        elif node_type == "interface_declaration":
            sym = _extract_interface(child, lines, module_name)
            if sym:
                symbols.append(sym)
                body = child.child_by_field_name("body")
                if body:
                    _walk_node(body, lines, module_name, symbols, imports, sym.name)

        elif node_type == "enum_declaration":
            sym = _extract_enum(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "record_declaration":
            sym = _extract_record(child, lines, module_name)
            if sym:
                symbols.append(sym)

        elif node_type == "method_declaration" and class_name:
            sym = _extract_method(child, lines, module_name, class_name)
            if sym:
                symbols.append(sym)

        elif node_type == "constructor_declaration" and class_name:
            sym = _extract_constructor(child, lines, module_name, class_name)
            if sym:
                symbols.append(sym)

        elif node_type == "field_declaration" and class_name:
            extracted = _extract_field(child, lines, module_name, class_name)
            symbols.extend(extracted)

        elif node_type == "import_declaration":
            edge = _extract_import(child)
            if edge:
                imports.append(edge)


def _extract_class(node, lines, module_name):
    """Extract a class declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    superclass = node.child_by_field_name("superclass")
    interfaces = node.child_by_field_name("interfaces")
    sig_parts = [f"class {name}"]
    if superclass:
        sig_parts.append(f"extends {_node_text(superclass)}")
    if interfaces:
        sig_parts.append(f"implements {_node_text(interfaces)}")

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="class",
        start_line=start,
        end_line=end,
        signature=" ".join(sig_parts),
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


def _extract_enum(node, lines, module_name):
    """Extract an enum declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="enum",
        start_line=start,
        end_line=end,
        signature=f"enum {name}",
    )


def _extract_record(node, lines, module_name):
    """Extract a record declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{name}",
        kind="record",
        start_line=start,
        end_line=end,
        signature=f"record {name}",
    )


def _extract_method(node, lines, module_name, class_name):
    """Extract a method declaration."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None
    name = _node_text(name_node)
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    return_type = node.child_by_field_name("type")
    params = node.child_by_field_name("parameters")
    ret_str = f"{_node_text(return_type)} " if return_type else ""
    sig = f"{ret_str}{name}{_node_text(params) if params else '()'}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{class_name}.{name}",
        kind="method",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_constructor(node, lines, module_name, class_name):
    """Extract a constructor declaration."""
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node else class_name
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1

    params = node.child_by_field_name("parameters")
    sig = f"{name}{_node_text(params) if params else '()'}"

    return SymbolEntry(
        name=name,
        qualified_name=f"{module_name}.{class_name}.{name}",
        kind="constructor",
        start_line=start,
        end_line=end,
        signature=sig,
    )


def _extract_field(node, lines, module_name, class_name):
    """Extract field declarations (may have multiple declarators)."""
    fields = []
    start = node.start_point[0] + 1
    end = node.end_point[0] + 1
    source = "\n".join(lines[start - 1 : end])

    for child in node.children:
        if child.type == "variable_declarator":
            name_node = child.child_by_field_name("name")
            if name_node:
                name = _node_text(name_node)
                fields.append(
                    SymbolEntry(
                        name=name,
                        qualified_name=f"{module_name}.{class_name}.{name}",
                        kind="field",
                        start_line=start,
                        end_line=end,
                        signature=source.strip().rstrip(";"),
                    )
                )
    return fields


def _extract_import(node):
    """Extract import edge from an import declaration."""
    text = _node_text(node).strip().rstrip(";")
    if text.startswith("import "):
        text = text[7:].strip()
        if text.startswith("static "):
            text = text[7:].strip()

        parts = text.rsplit(".", 1)
        if len(parts) == 2:
            module, name = parts
            return ImportEdge(
                target_module=module,
                import_names=[name] if name != "*" else [],
            )
        return ImportEdge(target_module=text, import_names=[])
    return None


def _extract_package(root_node):
    """Extract package name from root node."""
    for child in root_node.children:
        if child.type == "package_declaration":
            text = _node_text(child).strip().rstrip(";")
            if text.startswith("package "):
                return text[8:].strip()
    return None


def _node_text(node) -> str:
    """Get text content of a tree-sitter node."""
    if node is None:
        return ""
    if isinstance(node.text, bytes):
        return node.text.decode("utf-8")
    return str(node.text)


def _path_to_module(file_path: str) -> str:
    """Convert file path to package-like name."""
    path = file_path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if path.endswith(".java"):
        path = path[:-5]
    return path.replace("/", ".")
