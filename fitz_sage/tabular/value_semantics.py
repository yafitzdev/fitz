"""Shared value semantics for deterministic table queries."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from fitz_sage.core.identifiers import EXACT_IDENTIFIER_PATTERN


def normalize_table_value(value: Any) -> str:
    """Normalize a table value for deterministic predicate comparison."""
    text = "" if value is None else str(value)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def sortable_table_value(value: Any) -> float | None:
    """Return the shared numeric/date ordering value for a table cell."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(date.fromisoformat(text).toordinal())
    except ValueError:
        pass
    if EXACT_IDENTIFIER_PATTERN.fullmatch(text):
        return None
    match = re.fullmatch(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


__all__ = ["normalize_table_value", "sortable_table_value"]
