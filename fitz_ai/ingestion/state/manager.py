# fitz_ai/ingestion/state/manager.py
"""
State manager for incremental ingestion.

Handles loading, saving, and updating the .fitz/ingest.json state file.

Key responsibilities:
- Load/create state file
- Mark files as active (after successful ingestion)
- Mark files as deleted (when no longer on disk)
- Track config IDs (chunker_id, parser_id, embedding_id) per file
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fitz_ai.core.paths import FitzPaths
from fitz_ai.ingestion.state.schema import (
    EmbeddingConfig,
    FileEntry,
    FileStatus,
    IngestState,
    RootEntry,
)

logger = logging.getLogger(__name__)


class IngestStateManager:
    """
    Manages the ingestion state file.

    The state file tracks:
    - Which files have been ingested
    - Content hashes for change detection
    - Config IDs for re-chunking detection
    - Deletion tracking

    Usage:
        manager = IngestStateManager()
        manager.load()

        # After successful ingestion
        manager.mark_active(
            file_path="/path/to/file.md",
            root="/path/to",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        manager.save()
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        """
        Initialize the state manager.

        Args:
            path: Path to state file. Defaults to .fitz/ingest.json
        """
        self._path = path or FitzPaths.ingest_state()
        self._state: Optional[IngestState] = None
        self._dirty: bool = False

    @property
    def state(self) -> IngestState:
        """Get the current state. Raises if not loaded."""
        if self._state is None:
            raise RuntimeError("State not loaded. Call load() first.")
        return self._state

    def load(self) -> IngestState:
        """
        Load state from disk, or create new if not exists.

        Returns:
            The loaded or created state.
        """
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = IngestState.model_validate(data)
                logger.debug(f"Loaded ingest state from {self._path}")
            except Exception as e:
                logger.warning(f"Failed to load state, creating new: {e}")
                self._state = self._create_new_state()
                self._dirty = True
        else:
            self._state = self._create_new_state()
            self._dirty = True
            logger.debug("Created new ingest state")

        return self._state

    def save(self) -> None:
        """
        Save state to disk.

        Only writes if there are unsaved changes.
        """
        if self._state is None:
            return

        if not self._dirty:
            logger.debug("State unchanged, skipping save")
            return

        # Update timestamp
        self._state.updated_at = datetime.utcnow()

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically using temp file
        temp_path = self._path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(
                    self._state.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,
                )
            temp_path.replace(self._path)
            self._dirty = False
            logger.debug(f"Saved ingest state to {self._path}")
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _create_new_state(self) -> IngestState:
        """Create a new empty state."""
        return IngestState(
            project_id=str(uuid.uuid4()),
            updated_at=datetime.utcnow(),
        )

    def _ensure_root(self, root: str) -> RootEntry:
        """Ensure root entry exists, creating if necessary."""
        if root not in self.state.roots:
            self.state.roots[root] = RootEntry()
            self._dirty = True
        return self.state.roots[root]

    def mark_active(
        self,
        file_path: str,
        root: str,
        content_hash: str,
        ext: str,
        size_bytes: int,
        mtime_epoch: float,
        chunker_id: str,
        parser_id: str,
        embedding_id: str,
        enricher_id: Optional[str] = None,
        vector_db_id: Optional[str] = None,
        collection: str = "default",
    ) -> None:
        """
        Mark a file as active in state.

        Called after successful ingestion or when file is unchanged.

        Args:
            file_path: Absolute path to file.
            root: Absolute path to root directory.
            content_hash: SHA-256 content hash.
            ext: File extension (e.g., ".md").
            size_bytes: File size in bytes.
            mtime_epoch: File modification time as Unix epoch.
            chunker_id: Chunker ID used (e.g., "simple:1000:0").
            parser_id: Parser ID used (e.g., "md.v1").
            embedding_id: Embedding ID used (e.g., "cohere:embed-english-v3.0").
            enricher_id: Enricher ID used (e.g., "llm:gpt-4o-mini:v1"), None if not enriched.
            vector_db_id: Vector DB plugin used (e.g., "qdrant", "local_faiss").
            collection: Collection the file was ingested into.
        """
        root_entry = self._ensure_root(root)

        root_entry.files[file_path] = FileEntry(
            ext=ext,
            size_bytes=size_bytes,
            mtime_epoch=mtime_epoch,
            content_hash=content_hash,
            status=FileStatus.ACTIVE,
            ingested_at=datetime.utcnow(),
            chunker_id=chunker_id,
            parser_id=parser_id,
            embedding_id=embedding_id,
            vector_db_id=vector_db_id,
            enricher_id=enricher_id,
            collection=collection,
        )
        root_entry.last_scan_at = datetime.utcnow()
        self._dirty = True

    def mark_deleted(self, root: str, file_path: str) -> None:
        """
        Mark a file as deleted in state.

        Called when a file is no longer present on disk.

        Args:
            root: Absolute path to root directory.
            file_path: Absolute path to file.
        """
        root_entry = self.state.roots.get(root)
        if root_entry is None:
            return

        file_entry = root_entry.files.get(file_path)
        if file_entry is None:
            return

        if file_entry.status != FileStatus.DELETED:
            file_entry.status = FileStatus.DELETED
            self._dirty = True

    def get_active_paths(self, root: str) -> set[str]:
        """Get all active (non-deleted) file paths for a root."""
        return self.state.get_active_paths(root)

    def get_file_entry(self, root: str, file_path: str) -> Optional[FileEntry]:
        """Get file entry if it exists."""
        return self.state.get_file_entry(root, file_path)

    def set_embedding_config(
        self,
        provider: str,
        model: str,
        **kwargs,
    ) -> None:
        """
        Set the current embedding configuration.

        Args:
            provider: Embedding provider name.
            model: Model name.
            **kwargs: Additional config (dimension, etc.).
        """
        self.state.embedding = EmbeddingConfig.create(provider, model, **kwargs)
        self._dirty = True


__all__ = ["IngestStateManager"]
