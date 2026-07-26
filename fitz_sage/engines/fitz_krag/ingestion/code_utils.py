# fitz_sage/engines/fitz_krag/ingestion/code_utils.py
"""Shared helpers for code ingestion strategies and the import graph.

``node_text`` decodes a tree-sitter node; ``path_to_module`` derives a dotted
module/package name from a source path. Both were duplicated across the
per-language strategies (and the import-graph store) — this is their single home.
"""

from __future__ import annotations

from collections.abc import Iterable


def node_text(node) -> str:
    """Get the text content of a tree-sitter node (decoded to str)."""
    if node is None:
        return ""
    if isinstance(node.text, bytes):
        return node.text.decode("utf-8")
    return str(node.text)


def path_to_module(
    file_path: str, extensions: Iterable[str], strip_suffix: str | None = None
) -> str:
    """Convert a source path to a dotted module/package-like name.

    Strips a leading ``./``, the first matching extension, and an optional
    package-index suffix (e.g. ``/__init__`` for Python, ``/index`` for TS).
    """
    path = file_path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    for ext in extensions:
        if path.endswith(ext):
            path = path[: -len(ext)]
            break
    if strip_suffix and path.endswith(strip_suffix):
        path = path[: -len(strip_suffix)]
    return path.replace("/", ".")


def leading_comment_context(
    lines: list[str],
    start_line: int,
    *,
    include_annotations: bool = False,
) -> tuple[int, str]:
    """Return the contiguous doc-comment/annotation block before a declaration."""
    collected: list[tuple[int, str]] = []
    index = start_line - 2
    while index >= 0:
        stripped = lines[index].strip()
        if include_annotations and stripped.startswith("@"):
            collected.append((index, lines[index]))
            index -= 1
            continue
        if stripped.startswith("//"):
            collected.append((index, lines[index]))
            index -= 1
            continue
        if stripped.endswith("*/"):
            while index >= 0:
                collected.append((index, lines[index]))
                if "/*" in lines[index]:
                    index -= 1
                    break
                index -= 1
            continue
        break
    if not collected:
        return start_line, ""
    collected.sort(key=lambda item: item[0])
    return collected[0][0] + 1, "\n".join(line for _, line in collected)


def brace_block_end(lines: list[str], start_line: int) -> int:
    """Return the 1-based end line of a declaration's brace-delimited block."""
    depth = 0
    found_open = False
    for index in range(max(0, start_line - 1), len(lines)):
        for character in lines[index]:
            if character == "{":
                depth += 1
                found_open = True
            elif character == "}" and found_open:
                depth -= 1
        if found_open and depth <= 0:
            return index + 1
    return start_line
