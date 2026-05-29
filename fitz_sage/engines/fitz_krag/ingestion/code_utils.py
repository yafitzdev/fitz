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
