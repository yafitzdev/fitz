# core/vector_db/writer.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import VECTOR_DB
from fitz.core.models.chunk import Chunk

logger = get_logger(__name__)


def compute_chunk_hash(chunk: Chunk) -> str:
    """
    Stable hash for a chunk, used for dedupe/upsert keys.

    Hash includes:
    - doc_id
    - chunk_index
    - content
    """
    h = hashlib.sha256()
    h.update(chunk.doc_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(chunk.chunk_index).encode("utf-8"))
    h.update(b"\x00")
    h.update(chunk.content.encode("utf-8"))
    return h.hexdigest()


@dataclass
class VectorDBWriter:
    """
    Writes canonical Chunk objects into a Vector DB client/plugin.

    Payload contract (required by retrieval):
    - doc_id: str
    - chunk_index: int
    - content: str
    - metadata: dict
    """

    client: Any

    def upsert(
        self,
        collection: str,
        chunks: Iterable[Chunk],
        vectors: Iterable[list[float]],
    ) -> None:
        logger.info(f"{VECTOR_DB} Upserting chunks into collection='{collection}'")

        points = []

        for chunk, vector in zip(chunks, vectors):
            chunk_hash = compute_chunk_hash(chunk)

            payload: Dict[str, Any] = {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": dict(chunk.metadata or {}),
                "chunk_hash": chunk_hash,
            }

            points.append(
                {
                    "id": chunk.id,
                    "vector": vector,
                    "payload": payload,
                }
            )

        # Delegate to underlying client.
        # Vector DB plugins should adapt this to their backend.
        if hasattr(self.client, "upsert"):
            self.client.upsert(collection, points)
            return

        raise TypeError("VectorDBWriter client must expose an upsert(collection, points) method")
