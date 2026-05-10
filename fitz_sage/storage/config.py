# fitz_sage/storage/config.py
"""Configuration schema for unified PostgreSQL storage."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field

from fitz_sage.core.config_base import DatabaseConfig


class StorageMode(str, Enum):
    """Storage deployment mode."""

    LOCAL = "local"  # pgserver embedded PostgreSQL
    EXTERNAL = "external"  # External PostgreSQL connection string


class StorageConfig(DatabaseConfig):
    """
    Configuration for unified PostgreSQL storage.

    Inherits from DatabaseConfig for connection pooling and timeouts.

    Supports two modes:
    - LOCAL: Uses pgserver for embedded PostgreSQL (zero-friction local dev)
    - EXTERNAL: Uses user-provided PostgreSQL connection string (shared deployments)
    """

    mode: StorageMode = Field(
        default=StorageMode.LOCAL,
        description="Storage mode: 'local' (pgserver) or 'external' (connection_string)",
    )

    # Local mode settings
    data_dir: Optional[Path] = Field(
        default=None,
        description="Data directory for pgserver. None = use FitzPaths.pgdata()",
    )

    # Note: connection_string and pool settings inherited from DatabaseConfig

    def validate_mode(self) -> None:
        """Validate configuration based on mode."""
        if self.mode == StorageMode.EXTERNAL and not self.connection_string:
            raise ValueError("connection_string required for external mode")
