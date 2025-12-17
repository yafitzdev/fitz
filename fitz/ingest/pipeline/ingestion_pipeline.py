# ingest/pipeline/ingestion_pipeline.py
from __future__ import annotations

from fitz.logging.logger import get_logger
from fitz.logging.tags import PIPELINE
from fitz.vector_db.writer import VectorDBWriter
from fitz.ingest.chunking.engine import ChunkingEngine
from fitz.ingest.config.schema import IngestConfig
from fitz.ingest.ingestion.engine import IngestionEngine

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline:

        source
          -> ingestion (documents)
          -> chunking (chunks)
          -> embedding (vectors)   [must be injected upstream]
          -> vector db writer

    Architecture:
    - Construction happens via config-driven factories (for ingest/chunk).
    - Provider-specific embedding selection is NOT allowed here.
      Vectors must be provided by an injected embedder.
    """

    def __init__(
        self,
        *,
        config: IngestConfig,
        writer: VectorDBWriter,
        embedder: object,
    ) -> None:
        self.config = config
        self.writer = writer
        self.embedder = embedder

        self.ingester = IngestionEngine.from_config(config)
        self.chunker = ChunkingEngine.from_config(config.chunker)

        self.collection = config.collection

    def run(self, source: str) -> int:
        logger.info(f"{PIPELINE} Starting ingestion pipeline")

        total_written = 0

        for raw_doc in self.ingester.run(source):
            chunks = self.chunker.run(raw_doc)
            if not chunks:
                continue

            vectors = [self.embedder.embed(c.content) for c in chunks]

            self.writer.upsert(
                collection=self.collection,
                chunks=chunks,
                vectors=vectors,
            )
            total_written += len(chunks)

        logger.info(f"{PIPELINE} Ingestion finished, written={total_written}")
        return total_written
