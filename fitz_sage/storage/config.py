# fitz_sage/storage/config.py
"""Configuration schema for SQLite-backed storage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field

from fitz_sage.core.config_base import ValidationMixin


class StorageConfig(ValidationMixin):
    """Configuration for SQLite-backed storage.

    SQLite has no server, so the only knob is where the ``.db`` files
    live. ``None`` means ``FitzPaths.workspace() / "sqlite"``.
    """

    storage_path: Optional[Path] = Field(
        default=None,
        description="Directory holding per-collection .db files. "
        "None = use FitzPaths.workspace() / 'sqlite'.",
    )
