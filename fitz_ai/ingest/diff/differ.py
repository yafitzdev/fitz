# fitz_ai/ingest/diff/differ.py
"""
Diff computation for incremental ingestion.

Computes the action plan by comparing:
1. Scanned files from disk
2. State file (authoritative source for skip decisions)

Key design decision:
- State file (ingest.json) is the single source of truth
- If state says file was ingested with same hash → skip
- If state file is deleted → re-ingest everything (safe, idempotent upserts)

This module ONLY computes actions - it does NOT execute them.
Execution is handled by the executor.
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

    Used for both skip decisions and deletion detection.
    """

    def get_active_paths(self, root: str) -> Set[str]:
        """Get all active (non-deleted) file paths for a root."""
        ...

    def get_file_entry(self, root: str, file_path: str):
        """Get file entry if it exists. Returns None if not found."""
        ...

    def get_parser_id(self, ext: str) -> str:
        """Get parser ID for an extension."""
        ...

    def get_chunker_id(self, ext: str) -> str:
        """Get chunker ID for an extension."""
        ...

    def get_embedding_id(self) -> str:
        """Get the current embedding ID."""
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

    @classmethod
    def from_scanned(
            cls,
            scanned: ScannedFile,
            parser_id: str,
            chunker_id: str,
            embedding_id: str,
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
        )


@dataclass
class DiffResult:
    """
    Result of diff computation.

    Contains three lists of actions:
    - to_ingest: Files that need to be ingested (new or changed)
    - to_skip: Files that already exist in state (unchanged)
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


class Differ:
    """
    Computes diff between disk state and ingestion state.

    IMPORTANT: This class ONLY computes actions - it does NOT execute them.

    The diff algorithm:
    1. For each scanned file, check if state has entry with matching content_hash
    2. If state entry exists with same hash → skip
    3. If no state entry or hash differs → ingest
    4. For paths in state but not on disk → mark deleted

    Usage:
        differ = Differ(state_reader)
        result = differ.compute_diff(scan_result.files)

        # Result contains action plan, NOT executed actions
        for file in result.to_ingest:
            # These files need to be ingested
            pass
    """

    def __init__(self, state_reader: StateReader) -> None:
        """
        Initialize the differ.

        Args:
            state_reader: Reader for state (authoritative source)
        """
        self._state = state_reader

    def compute_diff(
            self,
            scanned_files: List[ScannedFile],
            force: bool = False,
            root: str | None = None,
    ) -> DiffResult:
        """
        Compute the diff action plan.

        Args:
            scanned_files: Files from the scanner
            force: If True, ingest all files regardless of state
            root: Root path for deletion detection. If None, extracted from scanned files.

        Returns:
            DiffResult with action plan
        """
        result = DiffResult()

        # Get embedding ID once
        embedding_id = self._state.get_embedding_id()

        # Track which paths we've seen for deletion detection
        seen_paths: Set[str] = set()
        detected_root: str | None = root

        for scanned in scanned_files:
            # Track root for deletion detection
            if detected_root is None:
                detected_root = scanned.root
            seen_paths.add(scanned.path)

            # Get config IDs for this file type
            parser_id = self._state.get_parser_id(scanned.ext)
            chunker_id = self._state.get_chunker_id(scanned.ext)

            candidate = FileCandidate.from_scanned(
                scanned,
                parser_id=parser_id,
                chunker_id=chunker_id,
                embedding_id=embedding_id,
            )

            if force:
                # Force mode: ingest everything
                result.to_ingest.append(candidate)
                continue

            # Check state file for existing entry with same hash
            existing_entry = self._state.get_file_entry(scanned.root, scanned.path)

            if existing_entry is not None and existing_entry.content_hash == scanned.content_hash:
                # Same hash in state → skip
                result.to_skip.append(candidate)
            else:
                # New file or hash changed → ingest
                result.to_ingest.append(candidate)

        # Deletion detection: compare state paths with scanned paths
        if detected_root is not None:
            previously_known = self._state.get_active_paths(detected_root)
            removed_paths = previously_known - seen_paths
            result.to_mark_deleted.extend(sorted(removed_paths))

        logger.info(f"Diff computed: {result.summary}")

        return result


def compute_diff(
        scanned_files: List[ScannedFile],
        state_reader: StateReader,
        force: bool = False,
        root: str | None = None,
) -> DiffResult:
    """
    Convenience function to compute diff.

    Args:
        scanned_files: Files from the scanner
        state_reader: Reader for state (authoritative source)
        force: If True, ingest all files regardless of state
        root: Root path for deletion detection. If None, extracted from scanned files.

    Returns:
        DiffResult with action plan
    """
    differ = Differ(state_reader)
    return differ.compute_diff(scanned_files, force=force, root=root)


__all__ = [
    "StateReader",
    "FileCandidate",
    "DiffResult",
    "Differ",
    "compute_diff",
]