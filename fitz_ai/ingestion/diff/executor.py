# fitz_ai/ingestion/diff/executor.py
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
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from fitz_ai.ingestion.chunking.router import ChunkingRouter
from fitz_ai.ingestion.diff.differ import Differ, FileCandidate
from fitz_ai.ingestion.diff.scanner import FileScanner
from fitz_ai.ingestion.hashing import compute_chunk_id
from fitz_ai.ingestion.state.manager import IngestStateManager

if TYPE_CHECKING:
    from fitz_ai.ingestion.enrichment.pipeline import EnrichmentPipeline

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

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts and return vectors."""
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
    hierarchy_summaries: int = 0  # Number of hierarchy summary chunks generated
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
            vector_db_id="qdrant",
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
        vector_db_id: Optional[str] = None,
        parser_id_func: Optional[callable] = None,
        enrichment_pipeline: Optional["EnrichmentPipeline"] = None,
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
            vector_db_id: Vector DB plugin name (e.g., "qdrant", "local_faiss").
            parser_id_func: Function to get parser_id for extension.
                           Defaults to "{ext}.v1" format.
            enrichment_pipeline: Optional unified enrichment pipeline.
                                Handles both chunk summaries and project artifacts.
        """
        self._state = state_manager
        self._vdb_writer = vector_db_writer
        self._embedder = embedder
        self._parser = parser
        self._router = chunking_router
        self._collection = collection
        self._embedding_id = embedding_id
        self._vector_db_id = vector_db_id
        self._get_parser_id = parser_id_func or self._default_parser_id
        self._enrichment_pipeline = enrichment_pipeline
        self._summary: Optional[IngestSummary] = None

    @staticmethod
    def _default_parser_id(ext: str) -> str:
        """Default parser ID function."""
        return f"{ext.lstrip('.')}.v1"

    @property
    def _enricher_id(self) -> Optional[str]:
        """Get the enricher ID if enrichment is enabled."""
        if self._enrichment_pipeline is None or not self._enrichment_pipeline.summaries_enabled:
            return None
        return self._enrichment_pipeline._enricher_id

    def run(
        self,
        source: str | Path,
        force: bool = False,
        on_progress: Optional[callable] = None,
        skip_artifacts: bool = False,
    ) -> IngestSummary:
        """
        Run the incremental ingestion pipeline.

        Args:
            source: Path to directory or file to ingest.
            force: If True, ingest everything regardless of state.
            on_progress: Optional callback(current, total, file_path) for progress updates.
            skip_artifacts: If True, skip artifact generation (caller handles separately).

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

        # 1.5 Generate and ingest artifacts (if pipeline provided and enabled)
        if (
            not skip_artifacts
            and self._enrichment_pipeline is not None
            and self._enrichment_pipeline.artifacts_enabled
        ):
            count, errors = self.ingest_artifacts()
            summary.artifacts_generated = count
            summary.errors += len(errors)
            summary.error_details.extend(errors)

        # 2. Compute diff (state + config aware)
        logger.info("Computing diff...")
        differ = Differ(
            state_reader=self._state.state,
            config_provider=self._router,
            parser_id_func=self._get_parser_id,
            embedding_id=self._embedding_id,
            vector_db_id=self._vector_db_id,
            collection=self._collection,
        )

        root_path = str(Path(source).resolve())
        diff = differ.compute_diff(scan_result.files, force=force, root=root_path)

        summary.skipped = len(diff.to_skip)

        # 3. Ingest files - three phases: prepare, summarize, embed
        summaries_enabled = (
            self._enrichment_pipeline and self._enrichment_pipeline.summaries_enabled
        )
        enrichment_status = "enabled" if summaries_enabled else "disabled"
        logger.info(f"Ingesting {len(diff.to_ingest)} files (summaries: {enrichment_status})...")

        total_to_ingest = len(diff.to_ingest)

        # Phase 1: Prepare all files (parse, chunk) - NO summarization yet
        all_prepared: List[tuple] = []  # (candidate, file_data)
        all_chunk_info: List[tuple] = (
            []
        )  # (content, file_path, content_hash) for batch summarization

        for i, candidate in enumerate(diff.to_ingest):
            if on_progress:
                on_progress(i, total_to_ingest, f"Preparing {Path(candidate.path).name}")

            try:
                file_data = self._prepare_file_no_enrich(candidate)
                if file_data is not None:
                    all_prepared.append((candidate, file_data))
                    # Collect chunk info for batch summarization
                    for chunk_data in file_data["chunk_data"]:
                        all_chunk_info.append(
                            (
                                chunk_data["chunk"].content,
                                candidate.path,
                                chunk_data["content_hash"],
                            )
                        )
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Prepare error: {candidate.path}: {e}")
                logger.warning(f"Failed to prepare {candidate.path}: {e}")

        # Phase 1.5: Batch summarize ALL chunks at once
        all_descriptions: List[str | None] = []
        if summaries_enabled and all_chunk_info:
            t0 = time.perf_counter()
            all_descriptions = self._enrichment_pipeline.summarize_chunks_batch(all_chunk_info)
            summarize_time = time.perf_counter() - t0
            summary.enriched = sum(1 for d in all_descriptions if d is not None)
            logger.info(f"Summarized {len(all_chunk_info)} chunks in {summarize_time:.2f}s")
        else:
            all_descriptions = [None] * len(all_chunk_info)

        # Apply descriptions back to chunk data and build texts to embed
        all_texts: List[str] = []
        desc_idx = 0
        for candidate, file_data in all_prepared:
            for chunk_data in file_data["chunk_data"]:
                description = all_descriptions[desc_idx]
                chunk_data["description"] = description
                # Embed description if available, otherwise content
                text_to_embed = description if description else chunk_data["chunk"].content
                all_texts.append(text_to_embed)
                desc_idx += 1

        # Phase 1.75: Hierarchy enrichment (generates group + corpus summaries)
        hierarchy_chunks: List = []
        if (
            self._enrichment_pipeline
            and hasattr(self._enrichment_pipeline, "_hierarchy_enricher")
            and self._enrichment_pipeline._hierarchy_enricher is not None
        ):
            # Collect all Chunk objects
            all_chunks = [
                chunk_data["chunk"]
                for _, file_data in all_prepared
                for chunk_data in file_data["chunk_data"]
            ]

            if all_chunks:
                t0 = time.perf_counter()
                enriched_chunks = self._enrichment_pipeline._hierarchy_enricher.enrich(all_chunks)
                hierarchy_time = time.perf_counter() - t0

                # Extract only the new hierarchy summary chunks
                hierarchy_chunks = [
                    c for c in enriched_chunks if c.metadata.get("is_hierarchy_summary", False)
                ]

                if hierarchy_chunks:
                    logger.info(
                        f"Generated {len(hierarchy_chunks)} hierarchy summaries "
                        f"in {hierarchy_time:.2f}s"
                    )
                    # Add hierarchy chunk texts for embedding
                    for hc in hierarchy_chunks:
                        all_texts.append(hc.content)

        # Phase 2: Batch embed ALL texts at once
        if all_texts:
            t0 = time.perf_counter()
            all_vectors = self._embedder.embed_batch(all_texts)
            embed_time = time.perf_counter() - t0
            logger.info(f"Embedded {len(all_texts)} chunks in {embed_time:.2f}s")
        else:
            all_vectors = []

        # Phase 3: Upsert to vector DB and update state
        vector_offset = 0
        for idx, (candidate, file_data) in enumerate(all_prepared):
            if on_progress:
                on_progress(idx + 1, len(all_prepared), f"Saving {Path(candidate.path).name}")

            try:
                num_chunks = len(file_data["chunk_data"])
                # Get vectors for this file
                file_vectors = all_vectors[vector_offset : vector_offset + num_chunks]
                vector_offset += num_chunks

                # Build and upsert points (defer persist to batch at end)
                self._upsert_file(candidate, file_data, file_vectors, defer_persist=True)
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
                    vector_db_id=self._vector_db_id,
                    collection=self._collection,
                )
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Upsert error: {candidate.path}: {e}")
                logger.warning(f"Failed to upsert {candidate.path}: {e}")

        # Phase 3.5: Upsert hierarchy summary chunks
        if hierarchy_chunks:
            try:
                # Hierarchy vectors are at the end of all_vectors
                hierarchy_vector_start = len(all_vectors) - len(hierarchy_chunks)
                hierarchy_vectors = all_vectors[hierarchy_vector_start:]

                hierarchy_points = []
                for hc, vec in zip(hierarchy_chunks, hierarchy_vectors):
                    hierarchy_points.append(
                        {
                            "id": hc.id,
                            "vector": vec,
                            "payload": {
                                "content": hc.content,
                                "doc_id": hc.doc_id,
                                "chunk_index": hc.chunk_index,
                                **hc.metadata,
                            },
                        }
                    )

                self._vdb_writer.upsert(self._collection, hierarchy_points, defer_persist=True)
                logger.info(f"Upserted {len(hierarchy_points)} hierarchy summary chunks")
                summary.hierarchy_summaries = len(hierarchy_chunks)
            except Exception as e:
                summary.errors += 1
                summary.error_details.append(f"Hierarchy upsert error: {e}")
                logger.warning(f"Failed to upsert hierarchy chunks: {e}")

        # Flush vector DB once after all upserts
        if hasattr(self._vdb_writer, "flush"):
            self._vdb_writer.flush()
        elif hasattr(self._vdb_writer, "_client") and hasattr(self._vdb_writer._client, "flush"):
            self._vdb_writer._client.flush()

        # Final progress update
        if on_progress and total_to_ingest > 0:
            on_progress(total_to_ingest, total_to_ingest, "Done")

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
                vector_db_id=self._vector_db_id,
                collection=self._collection,
            )

        # 5. Mark deletions
        for deleted_path in diff.to_mark_deleted:
            self._state.mark_deleted(root_path, deleted_path)
            summary.marked_deleted += 1

        # 6. Save state and enrichment cache
        self._state.save()
        if self._enrichment_pipeline:
            self._enrichment_pipeline.save_cache()

        summary.finished_at = datetime.utcnow()
        self._summary = None

        logger.info(f"Ingestion complete: {summary}")

        return summary

    def _prepare_file_no_enrich(self, candidate: FileCandidate) -> Optional[Dict]:
        """
        Prepare a file for ingestion (parse, chunk) without enrichment.

        Enrichment happens in a separate batch phase for efficiency.

        Args:
            candidate: File candidate with all metadata.

        Returns:
            file_data dict or None if file should be skipped.
        """
        # 1. Parse file
        text = self._parser.parse(candidate.path)
        if not text or not text.strip():
            logger.warning(f"Empty content from {candidate.path}, skipping")
            return None

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
            return None

        # 3. Prepare chunk data (no enrichment yet - that's batched later)
        chunk_data = []
        for chunk in chunks:
            chunk_id = compute_chunk_id(
                content_hash=candidate.content_hash,
                chunk_index=chunk.chunk_index,
                parser_id=candidate.parser_id,
                chunker_id=candidate.chunker_id,
                embedding_id=candidate.embedding_id,
            )

            chunk_content_hash = _hash_text(chunk.content)

            chunk_data.append(
                {
                    "chunk_id": chunk_id,
                    "chunk": chunk,
                    "content_hash": chunk_content_hash,
                    "description": None,  # Filled in by batch summarization
                }
            )

        return {"chunk_data": chunk_data}

    def _upsert_file(
        self,
        candidate: FileCandidate,
        file_data: Dict,
        vectors: List[List[float]],
        defer_persist: bool = False,
    ) -> None:
        """
        Build points and upsert to vector DB.

        Args:
            candidate: File candidate with all metadata.
            file_data: Prepared file data from _prepare_file_no_enrich.
            vectors: Embedding vectors for this file's chunks.
            defer_persist: If True, defer disk persistence (for batching).
        """
        chunk_data = file_data["chunk_data"]
        points = []

        for i, data in enumerate(chunk_data):
            chunk = data["chunk"]
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

            if data["description"] is not None:
                payload["description"] = data["description"]
                payload["enricher_id"] = self._enricher_id

            points.append(
                {
                    "id": data["chunk_id"],
                    "vector": vectors[i],
                    "payload": payload,
                }
            )

        self._vdb_writer.upsert(self._collection, points, defer_persist=defer_persist)
        logger.debug(f"Upserted {len(points)} chunks from {candidate.path}")

    def ingest_artifacts(
        self,
        on_progress: Optional[callable] = None,
    ) -> tuple[int, List[str]]:
        """
        Generate and ingest project artifacts.

        Artifacts are high-level summaries that provide context for code retrieval.
        They are stored with is_artifact=True and always retrieved with score=1.0.

        Args:
            on_progress: Optional callback(current, total, artifact_name) for progress.

        Returns:
            Tuple of (artifacts_generated, error_messages).
        """
        if self._enrichment_pipeline is None:
            return 0, []

        errors: List[str] = []
        logger.info("Generating project artifacts...")

        try:
            # Generate artifacts using the pipeline
            artifacts = self._enrichment_pipeline.generate_artifacts()

            if not artifacts:
                logger.info("No artifacts generated")
                return 0, []

            # Batch embed all artifacts at once
            total = len(artifacts)
            if on_progress:
                on_progress(0, total, "Embedding...")

            texts_to_embed = [artifact.content for artifact in artifacts]
            vectors = self._embedder.embed_batch(texts_to_embed)

            # Build points with vectors
            points = []
            for i, artifact in enumerate(artifacts):
                if on_progress:
                    on_progress(i + 1, total, artifact.artifact_type.value)

                # Create deterministic ID for artifact
                artifact_id = f"artifact:{artifact.artifact_type.value}:{self._collection}"

                # Build payload using artifact's to_payload method
                payload = artifact.to_payload()
                payload["collection"] = self._collection
                payload["embedding_id"] = self._embedding_id
                payload["ingested_at"] = datetime.utcnow().isoformat()

                points.append(
                    {
                        "id": artifact_id,
                        "vector": vectors[i],
                        "payload": payload,
                    }
                )

            # Upsert all artifacts
            self._vdb_writer.upsert(self._collection, points)
            logger.info(f"Ingested {len(artifacts)} artifacts")

            return len(artifacts), errors

        except Exception as e:
            errors.append(f"Artifact generation error: {e}")
            logger.warning(f"Failed to generate/ingest artifacts: {e}")
            return 0, errors


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
    vector_db_id: Optional[str] = None,
    parser_id_func: Optional[callable] = None,
    enrichment_pipeline: Optional["EnrichmentPipeline"] = None,
    force: bool = False,
    on_progress: Optional[callable] = None,
    skip_artifacts: bool = False,
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
        vector_db_id: Vector DB plugin name (e.g., "qdrant", "local_faiss").
        parser_id_func: Function to get parser_id for extension.
        enrichment_pipeline: Optional unified enrichment pipeline.
        force: If True, ingest everything regardless of state.
        on_progress: Optional callback(current, total, file_path) for progress updates.
        skip_artifacts: If True, skip artifact generation (caller handles separately).

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
        vector_db_id=vector_db_id,
        parser_id_func=parser_id_func,
        enrichment_pipeline=enrichment_pipeline,
    )
    return executor.run(source, force=force, on_progress=on_progress, skip_artifacts=skip_artifacts)


__all__ = [
    "VectorDBWriter",
    "Embedder",
    "Parser",
    "IngestSummary",
    "DiffIngestExecutor",
    "run_diff_ingest",
]
