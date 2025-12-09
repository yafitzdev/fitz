# src/fitz_rag/ingest/engine.py
"""
Generic ingestion engine for fitz-rag.

This engine:
- Accepts a Chunker
- Iterates a folder or list of files
- Produces Qdrant points
- Stores text + metadata into the vector DB
- Generates embeddings automatically via a provided embedder (optional)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Any

from qdrant_client import QdrantClient

from src.fitz_rag import ensure_collection


class IngestionEngine:
    """
    Ingest files into a Qdrant collection using:
    - Chunker (extract chunks from files)
    - Embedding client (optional)
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        vector_size: int,
        embedder: Optional[Any] = None,
        distance: str = "cosine",
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.vector_size = vector_size
        self.distance = distance

    # ---------------------------------------------------------
    # Ingest a path or list of paths
    # ---------------------------------------------------------
    def ingest_path(
        self,
        chunker,
        path: str | Path,
        glob_pattern: str = "**/*",
    ) -> None:
        """
        Ingests all files that match the glob pattern under the path.
        """
        path = Path(path)
        files = list(path.glob(glob_pattern))

        if not files:
            return

        # Ensure collection exists
        ensure_collection(
            client=self.client,
            name=self.collection,
            vector_size=self.vector_size,
        )

        for file in files:
            if file.is_file():
                self.ingest_file(chunker, file)

    # ---------------------------------------------------------
    # Ingest a single file
    # ---------------------------------------------------------
    def ingest_file(self, chunker, file_path: str | Path) -> None:
        chunks = chunker.chunk_file(str(file_path))

        if not chunks:
            return

        points = []

        for chunk in chunks:
            text = chunk.text
            meta = dict(chunk.metadata)
            meta["file"] = str(file_path)

            vector = (
                self.embedder.embed(text)
                if self.embedder is not None
                else None
            )

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
        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )
