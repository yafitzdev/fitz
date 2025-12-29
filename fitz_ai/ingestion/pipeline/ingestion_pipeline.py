# fitz_ai/ingestion/pipeline/ingestion_pipeline.py
from __future__ import annotations

from typing import TYPE_CHECKING

from fitz_ai.engines.classic_rag.config import IngestConfig
from fitz_ai.ingestion.chunking.engine import ChunkingEngine
from fitz_ai.ingestion.reader.engine import IngestionEngine
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE
from fitz_ai.vector_db.writer import VectorDBWriter

if TYPE_CHECKING:
    from fitz_ai.ingestion.enrichment.pipeline import EnrichmentPipeline

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline:

        source
          -> ingestion (documents)
          -> chunking (chunks)
          -> enrichment (optional: summaries + artifacts)
          -> embedding (vectors)   [must be injected upstream]
          -> vector db writer

    Architecture:
    - Construction happens via config-driven factories (for ingest/chunk).
    - Provider-specific embedding selection is NOT allowed here.
      Vectors must be provided by an injected embedder.
    - Enrichment is optional and adds summaries to chunks + generates artifacts.
    """

    def __init__(
        self,
        *,
        config: IngestConfig,
        writer: VectorDBWriter,
        embedder: object,
        enrichment: "EnrichmentPipeline | None" = None,
    ) -> None:
        self.config = config
        self.writer = writer
        self.embedder = embedder
        self.enrichment = enrichment

        self.ingester = IngestionEngine.from_config(config)
        self.chunker = ChunkingEngine.from_config(config.chunker)

        self.collection = config.collection

    def run(self, source: str) -> int:
        logger.info(f"{PIPELINE} Starting ingestion pipeline")

        # Collect all chunks first (needed for batch enrichment)
        all_chunks = []
        for raw_doc in self.ingester.run(source):
            chunks = self.chunker.run(raw_doc)
            if chunks:
                all_chunks.extend(chunks)

        if not all_chunks:
            logger.info(f"{PIPELINE} No chunks to process")
            return 0

        logger.info(f"{PIPELINE} Chunked {len(all_chunks)} chunks from source")

        # Enrichment stage (if enabled)
        artifacts = []
        if self.enrichment and self.enrichment.is_enabled:
            logger.info(f"{PIPELINE} Running enrichment pipeline")
            result = self.enrichment.enrich(all_chunks)
            all_chunks = result.chunks
            artifacts = result.artifacts
            logger.info(
                f"{PIPELINE} Enrichment complete: "
                f"{len(all_chunks)} chunks, {len(artifacts)} artifacts"
            )

        # Embed and store chunks
        vectors = [self.embedder.embed(c.content) for c in all_chunks]
        self.writer.upsert(
            collection=self.collection,
            chunks=all_chunks,
            vectors=vectors,
        )
        total_written = len(all_chunks)

        # Embed and store artifacts separately
        if artifacts:
            artifact_chunks = [a.to_chunk() for a in artifacts]
            artifact_vectors = [self.embedder.embed(c.content) for c in artifact_chunks]
            self.writer.upsert(
                collection=self.collection,
                chunks=artifact_chunks,
                vectors=artifact_vectors,
            )
            total_written += len(artifact_chunks)
            logger.info(f"{PIPELINE} Stored {len(artifact_chunks)} artifacts")

        logger.info(f"{PIPELINE} Ingestion finished, written={total_written}")
        return total_written
