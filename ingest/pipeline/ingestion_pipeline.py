# ingest/pipeline/ingestion_pipeline.py
from __future__ import annotations

from typing import Iterable

from ingest.config.schema import IngestConfig
from ingest.ingestion.engine import Ingester
from ingest.chunking.engine import ChunkingEngine

from core.vector_db.writer import VectorDBWriter
from core.logging.logger import get_logger
from core.logging.tags import PIPELINE

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline:

        source
          -> ingester
          -> chunker
          -> vector db writer

    Construction MUST happen via config-driven factories.
    """

    def __init__(
        self,
        *,
        config: IngestConfig,
        writer: VectorDBWriter,
    ) -> None:
        self.config = config
        self.writer = writer

        # -----------------------------------------------------
        # Build engines STRICTLY via config factories
        # -----------------------------------------------------
        self.ingester = Ingester.from_config(config)
        self.chunker = ChunkingEngine.from_config(config.chunker)

        self.collection = config.collection

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def run(self, source: str) -> int:
        logger.info(f"{PIPELINE} Starting ingestion pipeline")

        raw_docs = self.ingester.run(source)

        total_written = 0

        for raw_doc in raw_docs:
            chunks = self.chunker.run(raw_doc)
            if not chunks:
                continue

            written = self.writer.write(
                chunks=chunks,
                collection=self.collection,
            )
            total_written += written

        logger.info(f"{PIPELINE} Ingestion finished, written={total_written}")
        return total_written
