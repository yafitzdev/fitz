# fitz_ai/core/utils.py
"""
Core utilities shared across the Fitz codebase.

This module provides common helper functions used by multiple packages.
"""

from __future__ import annotations

import re
from typing import Any


def extract_path(
    data: Any, path: str, *, default: Any = None, strict: bool = True
) -> Any:
    """
    Extract a value from nested data using dot/bracket notation.

    This is the single implementation for path-based data extraction,
    used by both LLM runtime and Vector DB plugins.

    Args:
        data: The data structure to extract from (dict, list, or object)
        path: Dot-notation path with optional array indices
              Examples: "a.b", "items[0].text", "data[0]", "result.collections"
        default: Value to return if path not found (only used when strict=False)
        strict: If True (default), raises KeyError/IndexError on missing paths.
                If False, returns default value instead.

    Returns:
        Extracted value, or default if not found and strict=False

    Raises:
        KeyError: If path doesn't exist and strict=True
        IndexError: If array index is out of bounds and strict=True

    Examples:
        >>> extract_path({"a": {"b": 1}}, "a.b")
        1
        >>> extract_path({"items": [{"x": 1}]}, "items[0].x")
        1
        >>> extract_path({"data": [1, 2, 3]}, "data[0]")
        1
        >>> extract_path({"a": 1}, "b.c", default=None, strict=False)
        None
        >>> extract_path({"a": {"b": None}}, "a.b")
        None
    """
    if not path:
        return data

    # Split path into parts, handling both dots and brackets
    # "items[0].text" -> ["items", "0", "text"]
    # "result.collections" -> ["result", "collections"]
    parts = re.split(r"\.|\[|\]", path)
    parts = [p for p in parts if p]  # Remove empty strings

    current = data
    for part in parts:
        try:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, (list, tuple)):
                current = current[int(part)]
            elif hasattr(current, part):
                # Support attribute access for objects
                current = getattr(current, part)
            else:
                if strict:
                    raise KeyError(
                        f"Cannot traverse {type(current).__name__} with key {part!r}"
                    )
                return default
        except (KeyError, IndexError, TypeError):
            if strict:
                raise
            return default

    return current


def set_nested_path(data: dict, path: str, value: Any) -> None:
    """
    Set a value at a nested path, creating intermediate dicts as needed.

    Args:
        data: The dict to modify (modified in place)
        path: Dot-notation path (e.g., "options.temperature")
        value: Value to set

    Examples:
        >>> d = {}
        >>> set_nested_path(d, "a.b.c", 1)
        >>> d
        {'a': {'b': {'c': 1}}}
    """
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


__all__ = ["extract_path", "set_nested_path"]
