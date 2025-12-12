# src/fitz_ingest/pipeline/ingestion_pipeline.py
from __future__ import annotations

from dataclasses import dataclass

from ingest.ingester.engine import Ingester
from ingest.chunker.engine import ChunkingEngine

from core.vector_db.writer import VectorDBWriter
from core.logging.logger import get_logger
from core.logging.tags import INGEST, CHUNKING, VECTOR_DB, PIPELINE

logger = get_logger(__name__)


@dataclass
class IngestionPipeline:
    """
    Clean ingestion pipeline:

        Ingester → Chunker → VectorDBWriter

    Responsibilities:
    - run ingester
    - run chunker
    - pass chunks to writer (writer handles hash, uuid, dedupe, embedding, vectordb)
    """

    ingester: Ingester
    chunker: ChunkingEngine
    writer: VectorDBWriter
    collection: str

    def run(self, source: str) -> int:
        logger.info(f"{PIPELINE}{INGEST} Starting ingestion for source='{source}'")

        written = 0
        raw_docs = self.ingester.run(source)

        for raw_doc in raw_docs:
            doc_id = getattr(raw_doc, "id", None) or getattr(raw_doc, "path", None)
            logger.info(f"{INGEST} Processing document: {doc_id!r}")

            # Produce chunks
            chunks = list(self.chunker.run(raw_doc))
            if not chunks:
                logger.debug(f"{CHUNKING} No chunks produced for doc={doc_id!r}")
                continue

            # Writer handles:
            # - dedupe
            # - hashing
            # - embedding
            # - vector upsert
            count = self.writer.write(self.collection, chunks)

            logger.debug(f"{VECTOR_DB} Written {count} chunks for doc={doc_id!r}")
            written += count

        logger.info(
            f"{PIPELINE}{VECTOR_DB} Ingestion finished for source='{source}', "
            f"written_chunks={written}"
        )
        return written
