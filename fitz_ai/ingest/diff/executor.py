# fitz_ai/ingest/diff/executor.py
"""
Executor for incremental (diff) ingestion.

Orchestrates the full pipeline:
1. (Optional) Generate and ingest project artifacts
2. Scan files
3. Compute diff (detect content AND config changes)
4. Ingest new/changed files using file-type specific chunkers
5. (Optional) Enrich chunks with LLM-generated descriptions
6. Mark deletions
7. Update state
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from fitz_ai.ingest.chunking.router import ChunkingRouter
from fitz_ai.ingest.diff.differ import Differ, FileCandidate
from fitz_ai.ingest.diff.scanner import FileScanner
from fitz_ai.ingest.hashing import compute_chunk_id
from fitz_ai.ingest.state.manager import IngestStateManager

if TYPE_CHECKING:
    from fitz_ai.ingest.enrichment.router import EnrichmentRouter
    from fitz_ai.ingest.enrichment.artifacts.orchestrator import ArtifactOrchestrator

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorDBWriter(Protocol):
    """Protocol for writing to vector DB."""

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        """Upsert points into collection."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding text."""

    def embed(self, text: str) -> List[float]:
        """Embed text and return vector."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Protocol for parsing files."""

    def parse(self, path: str) -> str:
        """Parse a file and return its text content."""
        ...


@dataclass
class IngestSummary:
    """Summary of an ingestion run."""

    scanned: int = 0
    ingested: int = 0
    skipped: int = 0
    marked_deleted: int = 0
    errors: int = 0
    rechunked: int = 0  # Files re-ingested due to config change
    enriched: int = 0  # Chunks enriched with descriptions
    enrichment_cached: int = 0  # Chunks with cached descriptions
    artifacts_generated: int = 0  # Number of artifacts generated
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
        """Get summary string."""
        base = (
            f"scanned {self.scanned}, ingested {self.ingested}, "
            f"skipped {self.skipped}, marked_deleted {self.marked_deleted}, "
            f"errors {self.errors}"
        )
        if self.enriched > 0 or self.enrichment_cached > 0:
            base += f", enriched {self.enriched} (cached {self.enrichment_cached})"
        if self.artifacts_generated > 0:
            base += f", artifacts {self.artifacts_generated}"
        return base


class DiffIngestExecutor:
    """
    Executes the incremental ingestion pipeline.

    Uses ChunkingRouter to route files to type-specific chunkers.
    Optionally uses EnrichmentRouter to generate searchable descriptions.

    Usage:
        executor = DiffIngestExecutor(
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser=parser,
            chunking_router=router,
            collection="my_collection",
            embedding_id="cohere:embed-english-v3.0",
            enrichment_router=enrichment_router,  # Optional
        )

        summary = executor.run("/path/to/documents")
    """

    def __init__(
        self,
        *,
        state_manager: IngestStateManager,
        vector_db_writer: VectorDBWriter,
        embedder: Embedder,
        parser: Parser,
        chunking_router: ChunkingRouter,
        collection: str,
        embedding_id: str,
        parser_id_func: Optional[callable] = None,
        enrichment_router: Optional["EnrichmentRouter"] = None,
        artifact_orchestrator: Optional["ArtifactOrchestrator"] = None,
    ) -> None:
        """
        Initialize the executor.

        Args:
            state_manager: Manager for ingest state.
            vector_db_writer: Writer for upserting vectors.
            embedder: Embedder for creating vectors.
            parser: Parser for extracting text from files.
            chunking_router: Router for file-type specific chunking.
            collection: Vector DB collection name.
            embedding_id: Current embedding configuration ID.
            parser_id_func: Function to get parser_id for extension.
                           Defaults to "{ext}.v1" format.
            enrichment_router: Optional router for generating descriptions.
                              If provided, descriptions are embedded instead of content.
            artifact_orchestrator: Optional orchestrator for generating project artifacts.
                                  If provided, generates and ingests artifacts at start.
        """
        self._state = state_manager
        self._vdb_writer = vector_db_writer
        self._embedder = embedder
        self._parser = parser
        self._router = chunking_router
        self._collection = collection
        self._embedding_id = embedding_id
        self._get_parser_id = parser_id_func or self._default_parser_id
        self._enrichment_router = enrichment_router
        self._artifact_orchestrator = artifact_orchestrator
        self._summary: Optional[IngestSummary] = None  # Track current run stats

    @staticmethod
    def _default_parser_id(ext: str) -> str:
        """Default parser ID function."""
        return f"{ext.lstrip('.')}.v1"

    @property
    def _enricher_id(self) -> Optional[str]:
        """Get the enricher ID if enrichment is enabled."""
        if self._enrichment_router is None:
            return None
        return self._enrichment_router.enricher_id

    def run(
        self,
        source: str | Path,
        force: bool = False,
    ) -> IngestSummary:
        """
        Run the incremental ingestion pipeline.

        Args:
            source: Path to directory or file to ingest.
            force: If True, ingest everything regardless of state.

        Returns:
            IngestSummary with results.
        """
        summary = IngestSummary()
        self._summary = summary  # Store for access in _ingest_file

        # 1. Scan files
        logger.info(f"Scanning {source}...")
        scanner = FileScanner()
        scan_result = scanner.scan(source)
        summary.scanned = scan_result.total_scanned

        for path, error in scan_result.errors:
            summary.errors += 1
            summary.error_details.append(f"Scan error: {path}: {error}")

        # 1.5 Generate and ingest artifacts (if orchestrator provided)
        if self._artifact_orchestrator is not None:
            self._ingest_artifacts(summary)

        # 2. Compute diff (state + config aware)
        logger.info("Computing diff...")
        differ = Differ(
            state_reader=self._state.state,
            config_provider=self._router,
            parser_id_func=self._get_parser_id,
            embedding_id=self._embedding_id,
        )

        root_path = str(Path(source).resolve())
        diff = differ.compute_diff(scan_result.files, force=force, root=root_path)

        summary.skipped = len(diff.to_skip)

        # 3. Ingest files that need ingestion
        enrichment_status = "enabled" if self._enrichment_router else "disabled"
        logger.info(f"Ingesting {len(diff.to_ingest)} files (enrichment: {enrichment_status})...")

        for candidate in diff.to_ingest:
            try:
                self._ingest_file(candidate)
                summary.ingested += 1

                # Update state on success
                self._state.mark_active(
                    file_path=candidate.path,
                    root=candidate.root,
                    content_hash=candidate.content_hash,
                    ext=candidate.ext,
                    size_bytes=candidate.size_bytes,
                    mtime_epoch=candidate.mtime_epoch,
                    chunker_id=candidate.chunker_id,
                    parser_id=candidate.parser_id,
                    embedding_id=candidate.embedding_id,
                    enricher_id=self._enricher_id,
                )
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Ingest error: {candidate.path}: {e}")
                logger.warning(f"Failed to ingest {candidate.path}: {e}")

        # 4. Update state for skipped files (they're still active)
        for candidate in diff.to_skip:
            self._state.mark_active(
                file_path=candidate.path,
                root=candidate.root,
                content_hash=candidate.content_hash,
                ext=candidate.ext,
                size_bytes=candidate.size_bytes,
                mtime_epoch=candidate.mtime_epoch,
                chunker_id=candidate.chunker_id,
                parser_id=candidate.parser_id,
                embedding_id=candidate.embedding_id,
                enricher_id=self._enricher_id,
            )

        # 5. Mark deletions
        for deleted_path in diff.to_mark_deleted:
            self._state.mark_deleted(root_path, deleted_path)
            summary.marked_deleted += 1

        # 6. Save state and enrichment cache
        self._state.save()
        if self._enrichment_router:
            self._enrichment_router.save_cache()

        summary.finished_at = datetime.utcnow()
        self._summary = None

        logger.info(f"Ingestion complete: {summary}")

        return summary

    def _ingest_file(self, candidate: FileCandidate) -> None:
        """
        Ingest a single file.

        Args:
            candidate: File candidate with all metadata.
        """
        # 1. Parse file
        text = self._parser.parse(candidate.path)
        if not text or not text.strip():
            logger.warning(f"Empty content from {candidate.path}, skipping")
            return

        # 2. Get chunker for this file type and chunk
        chunker = self._router.get_chunker(candidate.ext)
        base_meta = {
            "source_file": candidate.path,
            "doc_id": Path(candidate.path).stem,
            "content_hash": candidate.content_hash,
            "parser_id": candidate.parser_id,
            "chunker_id": candidate.chunker_id,
        }
        chunks = chunker.chunk_text(text, base_meta)

        if not chunks:
            logger.warning(f"No chunks from {candidate.path}, skipping")
            return

        # 3. Build points with deterministic IDs
        points = []
        for chunk in chunks:
            chunk_id = compute_chunk_id(
                content_hash=candidate.content_hash,
                chunk_index=chunk.chunk_index,
                parser_id=candidate.parser_id,
                chunker_id=candidate.chunker_id,
                embedding_id=candidate.embedding_id,
            )

            # 3a. Enrichment (if enabled)
            description: Optional[str] = None
            if self._enrichment_router is not None:
                # Compute chunk-specific hash for caching
                chunk_content_hash = _hash_text(chunk.content)

                # Check if this is a cache hit by checking before/after
                cache_before = (chunk_content_hash, self._enrichment_router.enricher_id) in self._enrichment_router._cache

                description = self._enrichment_router.enrich(
                    content=chunk.content,
                    file_path=candidate.path,
                    content_hash=chunk_content_hash,
                )

                # Track enrichment stats
                if self._summary is not None:
                    if cache_before:
                        self._summary.enrichment_cached += 1
                    else:
                        self._summary.enriched += 1

            # 3b. Embed (description if enriched, content otherwise)
            text_to_embed = description if description else chunk.content
            vector = self._embedder.embed(text_to_embed)

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

            # Add enrichment data to payload if available
            if description is not None:
                payload["description"] = description
                payload["enricher_id"] = self._enrichment_router.enricher_id

            points.append(
                {
                    "id": chunk_id,
                    "vector": vector,
                    "payload": payload,
                }
            )

        # 4. Upsert to vector DB
        self._vdb_writer.upsert(self._collection, points)
        logger.debug(f"Upserted {len(points)} chunks from {candidate.path}")

    def _ingest_artifacts(self, summary: IngestSummary) -> None:
        """
        Generate and ingest project artifacts.

        Artifacts are high-level summaries that provide context for code retrieval.
        They are stored with is_artifact=True and always retrieved with score=1.0.

        Args:
            summary: IngestSummary to update with artifact stats.
        """
        if self._artifact_orchestrator is None:
            return

        logger.info("Generating project artifacts...")

        try:
            # Generate all artifacts
            artifacts = self._artifact_orchestrator.generate_all()

            if not artifacts:
                logger.info("No artifacts generated")
                return

            # Ingest each artifact
            points = []
            for artifact in artifacts:
                # Create deterministic ID for artifact
                artifact_id = f"artifact:{artifact.artifact_type.value}:{self._collection}"

                # Embed artifact content
                vector = self._embedder.embed(artifact.content)

                # Build payload using artifact's to_payload method
                payload = artifact.to_payload()
                payload["collection"] = self._collection
                payload["embedding_id"] = self._embedding_id
                payload["ingested_at"] = datetime.utcnow().isoformat()

                points.append({
                    "id": artifact_id,
                    "vector": vector,
                    "payload": payload,
                })

            # Upsert all artifacts
            self._vdb_writer.upsert(self._collection, points)
            summary.artifacts_generated = len(artifacts)

            logger.info(f"Ingested {len(artifacts)} artifacts")

        except Exception as e:
            summary.errors += 1
            summary.error_details.append(f"Artifact generation error: {e}")
            logger.warning(f"Failed to generate/ingest artifacts: {e}")


def _hash_text(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_diff_ingest(
    source: str | Path,
    *,
    state_manager: IngestStateManager,
    vector_db_writer: VectorDBWriter,
    embedder: Embedder,
    parser: Parser,
    chunking_router: ChunkingRouter,
    collection: str,
    embedding_id: str,
    parser_id_func: Optional[callable] = None,
    enrichment_router: Optional["EnrichmentRouter"] = None,
    artifact_orchestrator: Optional["ArtifactOrchestrator"] = None,
    force: bool = False,
) -> IngestSummary:
    """
    Convenience function to run diff ingestion.

    Args:
        source: Path to directory or file.
        state_manager: Manager for ingest state.
        vector_db_writer: Writer for upserting vectors.
        embedder: Embedder for creating vectors.
        parser: Parser for extracting text from files.
        chunking_router: Router for file-type specific chunking.
        collection: Vector DB collection name.
        embedding_id: Current embedding configuration ID.
        parser_id_func: Function to get parser_id for extension.
        enrichment_router: Optional router for generating descriptions.
        artifact_orchestrator: Optional orchestrator for generating project artifacts.
        force: If True, ingest everything regardless of state.

    Returns:
        IngestSummary with results.
    """
    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=vector_db_writer,
        embedder=embedder,
        parser=parser,
        chunking_router=chunking_router,
        collection=collection,
        embedding_id=embedding_id,
        parser_id_func=parser_id_func,
        enrichment_router=enrichment_router,
        artifact_orchestrator=artifact_orchestrator,
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
