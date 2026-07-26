"""Collection identity validation shared by storage and public interfaces."""

from __future__ import annotations

import re
from pathlib import Path

_COLLECTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_collection_name(name: str) -> str:
    """Return a valid collection name or raise without silently rewriting it."""
    if not isinstance(name, str) or not _COLLECTION_RE.fullmatch(name):
        raise ValueError(
            "Collection names must be 1-64 lowercase letters, numbers, underscores, "
            "or hyphens, and must start with a letter or number."
        )
    return name


def collection_name_from_path(source: Path) -> str:
    """Create a valid default name for a package-selected source path."""
    candidate = re.sub(r"[^a-z0-9_-]+", "_", source.resolve().name.strip().lower())
    candidate = candidate.strip("_-")[:64]
    return validate_collection_name(candidate or "default")


__all__ = ["collection_name_from_path", "validate_collection_name"]
