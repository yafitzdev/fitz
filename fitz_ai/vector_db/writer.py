# fitz_ai/vector_db/writer.py
"""
Vector database writer utilities.

Converts Chunk objects + vectors into the standard points format
that all vector database plugins accept.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Protocol, runtime_checkable

from fitz_ai.core.chunk import ChunkLike
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import VECTOR_DB

logger = get_logger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class VectorDBClient(Protocol):
    """Protocol for vector database clients."""

    def upsert(
        self, collection: str, points: List[Dict[str, Any]], defer_persist: bool = False
    ) -> None:
        """
        Insert or update points in collection.

        Args:
            collection: Target collection name
            points: List of point dicts with 'id', 'vector', 'payload'
            defer_persist: If True, defer disk writes (call flush() later for batching)
        """
        ...


# =============================================================================
# Utility Functions
# =============================================================================


def compute_chunk_hash(chunk: ChunkLike) -> str:
    """
    Compute a stable hash for a chunk.

    Hash includes doc_id, chunk_index, and content.
    Useful for deduplication and change detection.
    """
    h = hashlib.sha256()
    h.update(chunk.doc_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(chunk.chunk_index).encode("utf-8"))
    h.update(b"\x00")
    h.update(chunk.content.encode("utf-8"))
    return h.hexdigest()


def chunks_to_points(
    chunks: Iterable[ChunkLike],
    vectors: Iterable[List[float]],
) -> List[Dict[str, Any]]:
    """
    Convert chunks and vectors to the standard points format.

    This is the canonical transformation from (Chunk, vector) pairs
    to the format all vector databases accept.

    Args:
        chunks: Iterable of chunk-like objects
        vectors: Iterable of embedding vectors (same length as chunks)

    Returns:
        List of point dicts with 'id', 'vector', and 'payload' keys

    Example:
        >>> points = chunks_to_points(chunks, vectors)
        >>> vdb.upsert("collection", points)
    """
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

    return points


# =============================================================================
# VectorDBWriter Class
# =============================================================================


@dataclass
class VectorDBWriter:
    """
    Writes Chunk objects into a Vector DB client.

    This is a thin wrapper that:
    1. Converts chunks + vectors to points format
    2. Delegates to the underlying client

    For simple cases, you can use chunks_to_points() directly:

        points = chunks_to_points(chunks, vectors)
        vdb_client.upsert(collection, points)

    Payload contract (what retrieval expects):
        - doc_id: str
        - chunk_index: int
        - content: str
        - metadata: dict
        - chunk_hash: str (for deduplication)
    """

    client: VectorDBClient

    def upsert(
        self,
        collection: str,
        chunks: Iterable[ChunkLike],
        vectors: Iterable[List[float]],
    ) -> None:
        """
        Upsert chunks with their vectors into the vector database.

        Args:
            collection: Target collection name
            chunks: Chunk objects to store
            vectors: Embedding vectors (must match chunks order/length)
        """
        logger.info(f"{VECTOR_DB} Upserting chunks into collection='{collection}'")

        points = chunks_to_points(chunks, vectors)
        self.client.upsert(collection, points)

        logger.debug(f"{VECTOR_DB} Upserted {len(points)} points")
