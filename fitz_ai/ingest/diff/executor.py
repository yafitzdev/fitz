# fitz_ai/ingest/diff/executor.py
"""
Diff ingest executor.

Orchestrates the full incremental ingestion pipeline:
1. Scan files
2. Compute diff (via state file - authoritative source)
3. Ingest new/changed files
4. Mark deletions in vector DB
5. Update state
6. Report summary

Key responsibilities:
- Execute the action plan from Differ
- Handle errors gracefully (continue on failure)
- Update state only on success
- Mark deletions in vector DB

This is the ONLY place where writes happen (state + vector DB).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.base import ChunkerPlugin
from fitz_ai.ingest.hashing import compute_chunk_id
from fitz_ai.ingest.state import IngestStateManager

from .differ import DiffResult, FileCandidate
from .scanner import FileScanner

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorDBWriter(Protocol):
    """Protocol for writing to vector DB."""

    def upsert(
            self,
            collection: str,
            points: List[Dict[str, Any]],
    ) -> None:
        """Upsert points into collection."""
        ...

    def mark_deleted(
            self,
            collection: str,
            source_path: str,
    ) -> int:
        """
        Mark all vectors for a source path as deleted.

        Args:
            collection: Collection name
            source_path: File path to mark deleted

        Returns:
            Number of vectors marked as deleted
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text."""

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Protocol for parsing files to text."""

    def parse(self, path: str) -> str:
        """Parse a file and return its text content."""
        ...


@dataclass
class IngestSummary:
    """Summary of a diff ingest run."""

    scanned: int = 0
    ingested: int = 0
    skipped: int = 0
    marked_deleted: int = 0
    errors: int = 0
    error_details: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        if self.finished_at is None:
            return 0.0
        return (self.finished_at - self.started_at).total_seconds()

    def __str__(self) -> str:
        """Get summary string per spec §7.4."""
        return (
            f"scanned {self.scanned}, ingested {self.ingested}, "
            f"skipped {self.skipped}, marked_deleted {self.marked_deleted}, "
            f"errors {self.errors}"
        )


class DiffIngestExecutor:
    """
    Executes the incremental ingestion pipeline.

    This is the orchestrator that:
    1. Scans files
    2. Computes diff (using state file as authority)
    3. Ingests new/changed files
    4. Marks deletions
    5. Updates state

    Usage:
        executor = DiffIngestExecutor(
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser=parser,
            chunker=chunker_plugin,
            collection="my_collection",
        )

        summary = executor.run("/path/to/documents")
        print(summary)  # "scanned 10, ingested 3, skipped 7, ..."
    """

    def __init__(
            self,
            *,
            state_manager: IngestStateManager,
            vector_db_writer: VectorDBWriter,
            embedder: Embedder,
            parser: Parser,
            chunker: ChunkerPlugin,
            collection: str,
    ) -> None:
        """
        Initialize the executor.

        Args:
            state_manager: Manager for ingest state (authoritative source)
            vector_db_writer: Writer for upserting vectors
            embedder: Embedder for creating vectors
            parser: Parser for extracting text from files
            chunker: ChunkerPlugin for splitting text (must have chunk_text method)
            collection: Vector DB collection name
        """
        self._state = state_manager
        self._vdb_writer = vector_db_writer
        self._embedder = embedder
        self._parser = parser
        self._chunker = chunker
        self._collection = collection

    def run(
            self,
            source: str | Path,
            force: bool = False,
    ) -> IngestSummary:
        """
        Run the incremental ingestion pipeline.

        Args:
            source: Path to directory or file to ingest
            force: If True, ingest everything regardless of state

        Returns:
            IngestSummary with results
        """
        from .differ import Differ

        summary = IngestSummary()

        # 1. Scan files
        logger.info(f"Scanning {source}...")
        scanner = FileScanner()
        scan_result = scanner.scan(source)
        summary.scanned = scan_result.total_scanned

        # Add scan errors to summary
        for path, error in scan_result.errors:
            summary.errors += 1
            summary.error_details.append(f"Scan error: {path}: {error}")

        # 2. Compute diff (state file is authoritative)
        logger.info("Computing diff...")
        differ = Differ(state_reader=self._state.state)

        # Use resolved source path as root for deletion detection
        root_path = str(Path(source).resolve())
        diff = differ.compute_diff(scan_result.files, force=force, root=root_path)

        summary.skipped = len(diff.to_skip)

        # 3. Ingest files that need ingestion
        logger.info(f"Ingesting {len(diff.to_ingest)} files...")
        for candidate in diff.to_ingest:
            try:
                self._ingest_file(candidate)
                summary.ingested += 1

                # Update state only on success
                self._state.mark_active(
                    file_path=candidate.path,
                    root=candidate.root,
                    content_hash=candidate.content_hash,
                    ext=candidate.ext,
                    size_bytes=candidate.size_bytes,
                    mtime_epoch=candidate.mtime_epoch,
                )
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Ingest error: {candidate.path}: {e}")
                logger.warning(f"Failed to ingest {candidate.path}: {e}")
                # Do NOT update state for failed files (per spec §10)

        # 4. Update state for skipped files (they're still active)
        for candidate in diff.to_skip:
            self._state.mark_active(
                file_path=candidate.path,
                root=candidate.root,
                content_hash=candidate.content_hash,
                ext=candidate.ext,
                size_bytes=candidate.size_bytes,
                mtime_epoch=candidate.mtime_epoch,
            )

        # 5. Mark deletions in vector DB AND state
        logger.info(f"Marking {len(diff.to_mark_deleted)} files as deleted...")
        for path in diff.to_mark_deleted:
            try:
                count = self._vdb_writer.mark_deleted(self._collection, path)
                logger.debug(f"Marked {count} vectors as deleted for {path}")
                summary.marked_deleted += 1

                # Update state
                root = scan_result.root
                self._state.mark_deleted(path, root)
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Delete error: {path}: {e}")
                logger.warning(f"Failed to mark deleted {path}: {e}")

        # 6. Save state
        self._state.save()

        summary.finished_at = datetime.utcnow()
        logger.info(f"Ingestion complete: {summary}")

        return summary

    def _ingest_file(self, candidate: FileCandidate) -> None:
        """
        Ingest a single file.

        Parses, chunks, embeds, and upserts to vector DB.

        Args:
            candidate: File to ingest

        Raises:
            Exception: If any step fails
        """
        # 1. Parse file to text
        text = self._parser.parse(candidate.path)

        if not text or not text.strip():
            logger.warning(f"Empty content from {candidate.path}, skipping")
            return

        # 2. Chunk text
        base_meta = {
            "source_file": candidate.path,
            "doc_id": Path(candidate.path).stem,
            "content_hash": candidate.content_hash,
            "parser_id": candidate.parser_id,
            "chunker_id": candidate.chunker_id,
        }
        chunks = self._chunker.chunk_text(text, base_meta)

        if not chunks:
            logger.warning(f"No chunks from {candidate.path}, skipping")
            return

        # 3. Build points with deterministic IDs and full metadata
        points = []
        for chunk in chunks:
            # Compute deterministic chunk ID per spec §5.2
            chunk_id = compute_chunk_id(
                content_hash=candidate.content_hash,
                chunk_index=chunk.chunk_index,
                parser_id=candidate.parser_id,
                chunker_id=candidate.chunker_id,
                embedding_id=candidate.embedding_id,
            )

            # Embed the chunk
            vector = self._embedder.embed(chunk.content)

            # Build payload with required metadata per spec §5.1
            payload = {
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content_hash": candidate.content_hash,
                "source_path": candidate.path,
                "ext": candidate.ext,
                "chunk_text_hash": f"sha256:{_hash_text(chunk.content)}",
                "parser_id": candidate.parser_id,
                "chunker_id": candidate.chunker_id,
                "embedding_id": candidate.embedding_id,
                "is_deleted": False,
                "ingested_at": datetime.utcnow().isoformat(),
                "metadata": dict(chunk.metadata or {}),
            }

            points.append({
                "id": chunk_id,
                "vector": vector,
                "payload": payload,
            })

        # 4. Upsert to vector DB
        self._vdb_writer.upsert(self._collection, points)
        logger.debug(f"Upserted {len(points)} chunks from {candidate.path}")


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_diff_ingest(
        source: str | Path,
        *,
        state_manager: IngestStateManager,
        vector_db_writer: VectorDBWriter,
        embedder: Embedder,
        parser: Parser,
        chunker: ChunkerPlugin,
        collection: str,
        force: bool = False,
) -> IngestSummary:
    """
    Convenience function to run diff ingestion.

    Args:
        source: Path to directory or file
        state_manager: Manager for ingest state (authoritative source)
        vector_db_writer: Writer for upserting vectors
        embedder: Embedder for creating vectors
        parser: Parser for extracting text from files
        chunker: ChunkerPlugin for splitting text (must have chunk_text method)
        collection: Vector DB collection name
        force: If True, ingest everything regardless of state

    Returns:
        IngestSummary with results
    """
    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=vector_db_writer,
        embedder=embedder,
        parser=parser,
        chunker=chunker,
        collection=collection,
    )
    return executor.run(source, force=force)


__all__ = [
    "VectorDBWriter",
    "Embedder",
    "Parser",
    "IngestSummary",
    "DiffIngestExecutor",
    "run_diff_ingest",
]