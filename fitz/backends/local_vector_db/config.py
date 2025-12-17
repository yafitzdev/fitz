# fitz/backends/local_vector_db/config.py
"""
Configuration for local vector database backends.

Uses FitzPaths for default storage location.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def _default_vector_db_path() -> Path:
    """Get default vector DB path from FitzPaths."""
    from fitz.core.paths import FitzPaths

    return FitzPaths.vector_db()


class LocalVectorDBConfig(BaseModel):
    """
    Configuration for local vector database backends.

    Uses FitzPaths.vector_db() as default path, ensuring consistency
    across all components.
    """

    path: Path = Field(
        default_factory=_default_vector_db_path,
        description="Base directory for local vector database storage",
    )

    persist: bool = Field(
        default=True,
        description="Whether to persist the vector index to disk",
    )
