"""
Generic ingestion engine for fitz-ingest.

Handles:
- Chunking
- Optional embedding
- Upsert into Qdrant
- File iteration
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Any

from qdrant_client import QdrantClient

from fitz_ingest.vector_db.qdrant_utils import ensure_collection
from fitz_ingest.ingester.validation import (
    IngestionValidator,
    IngestionValidationError,
)

from fitz_rag.exceptions.retriever import EmbeddingError, VectorSearchError
from fitz_rag.exceptions.config import ConfigError


class IngestionEngine:
    """
    Ingest files into Qdrant using:
      - Chunker
      - Optional embedding client
      - Central validation (fitz_ingest.ingester.validation)
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector_size: int,
        embedder: Optional[Any] = None,
        distance: str = "cosine",
        validator: Optional[IngestionValidator] = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.vector_size = vector_size
        self.distance = distance
        self.validator = validator or IngestionValidator()

    # ---------------------------------------------------------
    # Ingest a directory tree
    # ---------------------------------------------------------
    def ingest_path(
        self,
        chunker,
        path: str | Path,
        glob_pattern: str = "**/*",
    ) -> None:
        try:
            path = Path(path)
            files = list(path.glob(glob_pattern))
        except Exception as e:
            raise ConfigError(f"Invalid ingestion path: {path}") from e

        if not files:
            return

        # Qdrant validation / ensure collection
        try:
            ensure_collection(
                client=self.client,
                name=self.collection,
                vector_size=self.vector_size,
            )
        except Exception as e:
            raise VectorSearchError(
                f"Failed ensuring collection '{self.collection}'"
            ) from e

        for file in files:
            if file.is_file():
                try:
                    self.ingest_file(chunker, file)
                except Exception as e:
                    raise VectorSearchError(
                        f"Failed ingesting file: {file}"
                    ) from e

    # ---------------------------------------------------------
    # Ingest a single file
    # ---------------------------------------------------------
    def ingest_file(self, chunker, file_path: str | Path) -> None:
        try:
            chunks = chunker.chunk_file(str(file_path))
        except Exception as e:
            raise ConfigError(f"Chunker failed on file: {file_path}") from e

        # No chunks -> nothing to write, not an error
        if not chunks:
            return

        # Validation BEFORE embedding / Qdrant
        try:
            if self.validator is not None:
                self.validator.validate_chunks(chunks, file_path)
        except IngestionValidationError as e:
            raise ConfigError(
                f"Ingestion validation failed for file '{file_path}': {e}"
            ) from e

        points = []

        for chunk in chunks:
            text = chunk.text
            meta = dict(chunk.metadata)
            meta["file"] = str(file_path)

            # Embedding step (optional)
            try:
                vector = (
                    self.embedder.embed(text)
                    if self.embedder is not None
                    else None
                )
            except Exception as e:
                raise EmbeddingError(
                    f"Failed embedding chunk from: {file_path}"
                ) from e

            point_id = str(uuid.uuid4())

            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "text": text,
                        **meta,
                    },
                }
            )

        # Upsert into Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
        except Exception as e:
            raise VectorSearchError(
                f"Failed upserting points into '{self.collection}'"
            ) from e
