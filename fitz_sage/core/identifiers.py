"""Literal structured-identifier parsing and matching."""

from __future__ import annotations

import re


EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])_?[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"
    r"(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*"
    r"(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9_])|"
    r"\b[A-Z]{2,}[A-Z0-9]*\d[A-Z0-9_-]*\b|"
    r"\b[A-Z]\d+\b"
)


def exact_identifiers(text: str) -> list[str]:
    """Extract distinct structured identifiers without normalizing them."""
    identifiers: list[str] = []
    seen: set[str] = set()
    for match in EXACT_IDENTIFIER_PATTERN.finditer(text):
        value = match.group(0).strip(".,;:()[]{}")
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        identifiers.append(value)
    return identifiers


def contains_exact_identifier(text: str, identifier: str) -> bool:
    """Return whether text contains the literal identifier as one complete token."""
    if not identifier:
        return False
    escaped = re.escape(identifier)
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_-]|\.[A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    )


__all__ = ["EXACT_IDENTIFIER_PATTERN", "contains_exact_identifier", "exact_identifiers"]
