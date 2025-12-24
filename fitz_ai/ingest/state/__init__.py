# fitz_ai/ingest/state/__init__.py
"""
State management for incremental ingestion.

This package handles the .fitz/ingest.json state file that tracks:
- Which files have been ingested
- Content hashes for change detection
- Config IDs (chunker_id, parser_id, embedding_id) for re-chunking detection
- Deletion tracking

Key exports:
- IngestStateManager: Load/save state, update entries
- IngestState: Root Pydantic model
- FileEntry: Per-file tracking with config IDs
"""

from fitz_ai.ingest.state.manager import IngestStateManager
from fitz_ai.ingest.state.schema import (
    EmbeddingConfig,
    FileEntry,
    FileStatus,
    IngestState,
    RootEntry,
)

__all__ = [
    # Manager
    "IngestStateManager",
    # Schema
    "IngestState",
    "RootEntry",
    "FileEntry",
    "FileStatus",
    "EmbeddingConfig",
]
