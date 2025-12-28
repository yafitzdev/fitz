# fitz_ai/ingest/state/schema.py
"""
State schema for incremental ingestion.

The state file (.fitz/ingest.json) tracks:
- Which files have been ingested
- Content hashes for change detection
- Config IDs (chunker_id, parser_id, embedding_id) for re-chunking detection

When any config ID changes, the file will be re-ingested on next run.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class FileStatus(str, Enum):
    """Status of a tracked file."""

    ACTIVE = "active"
    DELETED = "deleted"


class FileEntry(BaseModel):
    """
    State entry for a single file.

    Tracks everything needed to detect if re-ingestion is required:
    - content_hash: Detects content changes
    - chunker_id: Detects chunking config changes
    - parser_id: Detects parser config changes
    - embedding_id: Detects embedding config changes
    - enricher_id: Detects enrichment config changes (optional)
    """

    model_config = ConfigDict(extra="forbid")

    content_hash: str = Field(..., description="SHA-256 hash of file content")
    ext: str = Field(..., description="File extension (e.g., '.md')")
    size_bytes: int = Field(..., description="File size in bytes")
    mtime_epoch: float = Field(..., description="Modification time as Unix epoch")
    status: FileStatus = Field(default=FileStatus.ACTIVE, description="File status")
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow, description="When file was last ingested"
    )

    # Config IDs - stored to detect config changes
    chunker_id: str = Field(..., description="Chunker ID used (e.g., 'simple:1000:0')")
    parser_id: str = Field(..., description="Parser ID used (e.g., 'md.v1')")
    embedding_id: str = Field(
        ..., description="Embedding ID used (e.g., 'cohere:embed-english-v3.0')"
    )
    enricher_id: Optional[str] = Field(
        default=None,
        description="Enricher ID used (e.g., 'llm:gpt-4o-mini:v1'), None if not enriched",
    )
    collection: str = Field(
        default="default",
        description="Collection the file was ingested into",
    )

    def is_active(self) -> bool:
        """Check if file is active (not deleted)."""
        return self.status == FileStatus.ACTIVE


class RootEntry(BaseModel):
    """
    State entry for a root directory.

    A root is a directory that was ingested. Files are tracked relative to roots.
    """

    model_config = ConfigDict(extra="forbid")

    files: Dict[str, FileEntry] = Field(
        default_factory=dict, description="Files keyed by absolute path"
    )
    last_scan_at: datetime = Field(
        default_factory=datetime.utcnow, description="When root was last scanned"
    )


class EmbeddingConfig(BaseModel):
    """Current embedding configuration."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., description="Embedding provider (e.g., 'cohere')")
    model: str = Field(..., description="Model name (e.g., 'embed-english-v3.0')")
    dimension: Optional[int] = Field(default=None, description="Vector dimension")
    id: str = Field(..., description="Composite ID: provider:model")

    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        dimension: Optional[int] = None,
    ) -> "EmbeddingConfig":
        """Create an EmbeddingConfig with auto-generated ID."""
        return cls(
            provider=provider,
            model=model,
            dimension=dimension,
            id=f"{provider}:{model}",
        )


class IngestState(BaseModel):
    """
    Root state model for .fitz/ingest.json.

    This is the single source of truth for ingestion state.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, description="Schema version for migrations")
    project_id: str = Field(..., description="UUID for this project")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    roots: Dict[str, RootEntry] = Field(
        default_factory=dict, description="Tracked roots keyed by abs path"
    )
    embedding: Optional[EmbeddingConfig] = Field(
        default=None, description="Current embedding config"
    )

    def get_file_entry(self, root: str, file_path: str) -> Optional[FileEntry]:
        """Get file entry if it exists."""
        root_entry = self.roots.get(root)
        if root_entry is None:
            return None
        return root_entry.files.get(file_path)

    def get_active_paths(self, root: str) -> set[str]:
        """Get all active (non-deleted) file paths for a root."""
        root_entry = self.roots.get(root)
        if root_entry is None:
            return set()
        return {path for path, entry in root_entry.files.items() if entry.is_active()}


__all__ = [
    "FileStatus",
    "FileEntry",
    "RootEntry",
    "EmbeddingConfig",
    "IngestState",
]
