# fitz_ai/ingestion/diff/differ.py
"""
Diff computation for incremental ingestion.

Computes the action plan by comparing:
1. Scanned files from disk
2. State file (authoritative source for skip decisions)
3. Current chunking configuration (chunker_id per extension)

Re-ingestion is triggered when:
- File content changes (content_hash differs)
- Chunking config changes (chunker_id differs)
- Embedding config changes (embedding_id differs)
- Parser config changes (parser_id differs)

This module ONLY computes actions - it does NOT execute them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Protocol, Set, runtime_checkable

from .scanner import ScannedFile

logger = logging.getLogger(__name__)


@runtime_checkable
class StateReader(Protocol):
    """
    Protocol for reading ingestion state.

    Used for skip decisions and deletion detection.
    """

    def get_active_paths(self, root: str) -> Set[str]:
        """Get all active (non-deleted) file paths for a root."""
        ...

    def get_file_entry(self, root: str, file_path: str):
        """Get file entry if it exists. Returns None if not found."""
        ...


@runtime_checkable
class ConfigProvider(Protocol):
    """
    Protocol for providing current configuration IDs.

    The ChunkingRouter implements this via get_chunker_id().
    """

    def get_chunker_id(self, ext: str) -> str:
        """Get the current chunker_id for an extension."""
        ...


@dataclass(frozen=True)
class FileCandidate:
    """
    A file candidate for ingestion.

    Contains all information needed to ingest or skip a file.
    """

    path: str  # Absolute path
    root: str  # Absolute path of scan root
    ext: str  # Extension (e.g., ".md")
    size_bytes: int  # File size
    mtime_epoch: float  # Modification time
    content_hash: str  # SHA-256 content hash
    parser_id: str  # Parser ID for this file type
    chunker_id: str  # Chunker ID for this file type
    embedding_id: str  # Embedding ID
    vector_db_id: str | None = None  # Vector DB plugin name

    @classmethod
    def from_scanned(
        cls,
        scanned: ScannedFile,
        parser_id: str,
        chunker_id: str,
        embedding_id: str,
        vector_db_id: str | None = None,
    ) -> "FileCandidate":
        """Create from a ScannedFile with config IDs."""
        return cls(
            path=scanned.path,
            root=scanned.root,
            ext=scanned.ext,
            size_bytes=scanned.size_bytes,
            mtime_epoch=scanned.mtime_epoch,
            content_hash=scanned.content_hash,
            parser_id=parser_id,
            chunker_id=chunker_id,
            embedding_id=embedding_id,
            vector_db_id=vector_db_id,
        )


@dataclass
class DiffResult:
    """
    Result of diff computation.

    Contains three lists of actions:
    - to_ingest: Files that need to be ingested (new, changed, or config changed)
    - to_skip: Files that are unchanged (same content AND same config)
    - to_mark_deleted: File paths that no longer exist on disk
    """

    to_ingest: List[FileCandidate] = field(default_factory=list)
    to_skip: List[FileCandidate] = field(default_factory=list)
    to_mark_deleted: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"ingest={len(self.to_ingest)}, "
            f"skip={len(self.to_skip)}, "
            f"delete={len(self.to_mark_deleted)}"
        )


@dataclass
class ReingestReason:
    """Reason why a file needs re-ingestion."""

    content_changed: bool = False
    chunker_changed: bool = False
    parser_changed: bool = False
    embedding_changed: bool = False
    vector_db_changed: bool = False
    collection_changed: bool = False
    is_new: bool = False

    @property
    def needs_reingest(self) -> bool:
        return (
            self.content_changed
            or self.chunker_changed
            or self.parser_changed
            or self.embedding_changed
            or self.vector_db_changed
            or self.collection_changed
            or self.is_new
        )

    def __str__(self) -> str:
        reasons = []
        if self.is_new:
            reasons.append("new")
        if self.content_changed:
            reasons.append("content_changed")
        if self.chunker_changed:
            reasons.append("chunker_changed")
        if self.parser_changed:
            reasons.append("parser_changed")
        if self.embedding_changed:
            reasons.append("embedding_changed")
        if self.vector_db_changed:
            reasons.append("vector_db_changed")
        if self.collection_changed:
            reasons.append("collection_changed")
        return ", ".join(reasons) if reasons else "none"


class Differ:
    """
    Computes diff between disk state and ingestion state.

    The diff algorithm checks for:
    1. New files (not in state)
    2. Content changes (content_hash differs)
    3. Config changes (chunker_id, parser_id, embedding_id, vector_db_id, or collection differs)

    Usage:
        differ = Differ(
            state_reader=state_manager.state,
            config_provider=chunking_router,
            parser_id_func=lambda ext: f"{ext.lstrip('.')}.v1",
            embedding_id="cohere:embed-english-v3.0",
            vector_db_id="qdrant",
            collection="my_collection",
        )
        result = differ.compute_diff(scan_result.files)
    """

    def __init__(
        self,
        state_reader: StateReader,
        config_provider: ConfigProvider,
        parser_id_func: callable,
        embedding_id: str,
        vector_db_id: str | None = None,
        collection: str = "default",
    ) -> None:
        """
        Initialize the differ.

        Args:
            state_reader: Reader for ingestion state.
            config_provider: Provider for current config IDs (e.g., ChunkingRouter).
            parser_id_func: Function to get parser_id for an extension.
            embedding_id: Current embedding configuration ID.
            vector_db_id: Current vector DB plugin name (e.g., "qdrant", "local_faiss").
            collection: Target collection for ingestion.
        """
        self._state = state_reader
        self._config = config_provider
        self._get_parser_id = parser_id_func
        self._embedding_id = embedding_id
        self._vector_db_id = vector_db_id
        self._collection = collection

    def _check_reingest_reason(
        self,
        scanned: ScannedFile,
        current_chunker_id: str,
        current_parser_id: str,
    ) -> ReingestReason:
        """
        Check why a file might need re-ingestion.

        Args:
            scanned: The scanned file from disk.
            current_chunker_id: Current chunker_id from config.
            current_parser_id: Current parser_id from config.

        Returns:
            ReingestReason with flags for what changed.
        """
        existing = self._state.get_file_entry(scanned.root, scanned.path)

        if existing is None:
            return ReingestReason(is_new=True)

        reason = ReingestReason()

        # Check content change
        if existing.content_hash != scanned.content_hash:
            reason.content_changed = True

        # Check chunker config change
        existing_chunker_id = getattr(existing, "chunker_id", None)
        if existing_chunker_id != current_chunker_id:
            reason.chunker_changed = True

        # Check parser config change
        existing_parser_id = getattr(existing, "parser_id", None)
        if existing_parser_id != current_parser_id:
            reason.parser_changed = True

        # Check embedding config change
        existing_embedding_id = getattr(existing, "embedding_id", None)
        if existing_embedding_id != self._embedding_id:
            reason.embedding_changed = True

        # Check vector DB change (different DB = re-ingest, data is elsewhere)
        if self._vector_db_id is not None:
            existing_vector_db_id = getattr(existing, "vector_db_id", None)
            if existing_vector_db_id != self._vector_db_id:
                reason.vector_db_changed = True

        # Check collection change (different collection = re-ingest)
        existing_collection = getattr(existing, "collection", "default")
        if existing_collection != self._collection:
            reason.collection_changed = True

        return reason

    def compute_diff(
        self,
        scanned_files: List[ScannedFile],
        force: bool = False,
        root: str | None = None,
    ) -> DiffResult:
        """
        Compute the diff action plan.

        Args:
            scanned_files: Files from the scanner.
            force: If True, ingest all files regardless of state.
            root: Root path for deletion detection.

        Returns:
            DiffResult with action plan.
        """
        result = DiffResult()

        seen_paths: Set[str] = set()
        detected_root: str | None = root

        for scanned in scanned_files:
            if detected_root is None:
                detected_root = scanned.root
            seen_paths.add(scanned.path)

            # Get current config IDs
            current_chunker_id = self._config.get_chunker_id(scanned.ext)
            current_parser_id = self._get_parser_id(scanned.ext)

            candidate = FileCandidate.from_scanned(
                scanned,
                parser_id=current_parser_id,
                chunker_id=current_chunker_id,
                embedding_id=self._embedding_id,
                vector_db_id=self._vector_db_id,
            )

            if force:
                result.to_ingest.append(candidate)
                continue

            # Check if re-ingestion is needed
            reason = self._check_reingest_reason(scanned, current_chunker_id, current_parser_id)

            if reason.needs_reingest:
                logger.debug(f"Re-ingest {scanned.path}: {reason}")
                result.to_ingest.append(candidate)
            else:
                result.to_skip.append(candidate)

        # Deletion detection
        if detected_root is not None:
            previously_known = self._state.get_active_paths(detected_root)
            removed_paths = previously_known - seen_paths
            result.to_mark_deleted.extend(sorted(removed_paths))

        logger.info(f"Diff computed: {result.summary}")

        return result


def compute_diff(
    scanned_files: List[ScannedFile],
    state_reader: StateReader,
    config_provider: ConfigProvider,
    parser_id_func: callable,
    embedding_id: str,
    force: bool = False,
    root: str | None = None,
    vector_db_id: str | None = None,
    collection: str = "default",
) -> DiffResult:
    """
    Convenience function to compute diff.

    Args:
        scanned_files: Files from the scanner.
        state_reader: Reader for ingestion state.
        config_provider: Provider for current config IDs.
        parser_id_func: Function to get parser_id for an extension.
        embedding_id: Current embedding configuration ID.
        force: If True, ingest all files regardless of state.
        root: Root path for deletion detection.
        vector_db_id: Current vector DB plugin name (e.g., "qdrant", "local_faiss").
        collection: Target collection for ingestion.

    Returns:
        DiffResult with action plan.
    """
    differ = Differ(
        state_reader=state_reader,
        config_provider=config_provider,
        parser_id_func=parser_id_func,
        embedding_id=embedding_id,
        vector_db_id=vector_db_id,
        collection=collection,
    )
    return differ.compute_diff(scanned_files, force=force, root=root)


__all__ = [
    "StateReader",
    "ConfigProvider",
    "FileCandidate",
    "DiffResult",
    "ReingestReason",
    "Differ",
    "compute_diff",
]
