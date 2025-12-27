# fitz_ai/map/embeddings.py
"""
Embedding fetching and caching utilities for knowledge map.

Handles fetching chunk embeddings from vector DB and converting to float16 for storage.
"""

from __future__ import annotations

import logging
from typing import Any, List, Set, Tuple

import numpy as np

from fitz_ai.map.models import ChunkEmbedding

logger = logging.getLogger(__name__)


def fetch_all_chunk_ids(vector_db: Any, collection: str) -> Set[str]:
    """
    Fetch all chunk IDs from vector DB via scroll.

    Works with FAISS and other DBs that support scroll.

    Args:
        vector_db: Vector DB plugin instance.
        collection: Collection name.

    Returns:
        Set of all chunk IDs in the collection.
    """
    chunk_ids: Set[str] = set()
    offset = 0
    batch_size = 100

    while True:
        records, next_offset = vector_db.scroll(
            collection=collection,
            limit=batch_size,
            offset=offset,
        )

        for record in records:
            # Handle both _FaissRecord objects and dicts
            if hasattr(record, "id"):
                chunk_ids.add(record.id)
            else:
                chunk_ids.add(record["id"])

        if next_offset is None:
            break
        offset = next_offset

    logger.debug(f"Found {len(chunk_ids)} chunks in collection '{collection}'")
    return chunk_ids


def fetch_chunk_embeddings(
    vector_db: Any,
    collection: str,
    chunk_ids: Set[str] | None = None,
    batch_size: int = 100,
) -> List[ChunkEmbedding]:
    """
    Fetch embeddings for chunks from vector DB.

    Uses scroll_with_vectors for FAISS which returns vectors directly.

    Args:
        vector_db: Vector DB plugin instance.
        collection: Collection name.
        chunk_ids: Optional set of specific chunk IDs to fetch.
                   If None, fetches all chunks.
        batch_size: Number of records to fetch per batch.

    Returns:
        List of ChunkEmbedding objects with float16 vectors.
    """
    embeddings: List[ChunkEmbedding] = []
    offset = 0

    # Check if vector DB supports scroll_with_vectors
    if not hasattr(vector_db, "scroll_with_vectors"):
        raise NotImplementedError(
            f"Vector DB {type(vector_db).__name__} does not support scroll_with_vectors. "
            "Knowledge map currently only supports FAISS backend."
        )

    while True:
        records, next_offset = vector_db.scroll_with_vectors(
            collection=collection,
            limit=batch_size,
            offset=offset,
        )

        for record in records:
            record_id = record["id"]

            # Filter to specific chunk_ids if provided
            if chunk_ids is not None and record_id not in chunk_ids:
                continue

            payload = record["payload"]
            vector = record["vector"]

            # Compress to float16
            compressed = compress_to_float16(vector)

            # Extract metadata
            content = payload.get("content", "")
            doc_id = payload.get("doc_id", "unknown")
            chunk_index = payload.get("chunk_index", 0)
            metadata = payload.get("metadata", {})

            # Create label from content preview
            label = _create_label(content, metadata)
            content_preview = content[:200] if content else ""

            embeddings.append(
                ChunkEmbedding(
                    chunk_id=record_id,
                    doc_id=doc_id,
                    label=label,
                    embedding=compressed,
                    chunk_index=chunk_index,
                    content_preview=content_preview,
                    metadata=metadata,
                )
            )

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        f"Fetched {len(embeddings)} chunk embeddings from collection '{collection}'"
    )
    return embeddings


def _create_label(content: str, metadata: dict) -> str:
    """Create a display label from content or metadata."""
    # Try to get title from metadata
    title = metadata.get("title") or metadata.get("summary")
    if title:
        return title[:50]

    # Fall back to first line of content
    if content:
        first_line = content.split("\n")[0].strip()
        if first_line:
            return first_line[:50]

    return "Untitled"


def compress_to_float16(embedding: List[float]) -> List[float]:
    """
    Compress embedding to float16 for storage.

    Reduces storage by ~50% with minimal accuracy loss.

    Args:
        embedding: Original embedding vector (float32/64).

    Returns:
        Compressed embedding as list of floats (stored as float16 precision).
    """
    arr = np.array(embedding, dtype=np.float16)
    return arr.tolist()


def embeddings_to_matrix(chunks: List[ChunkEmbedding]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert list of ChunkEmbedding to numpy matrix.

    Args:
        chunks: List of ChunkEmbedding objects.

    Returns:
        Tuple of (embedding_matrix, chunk_ids)
        - embedding_matrix: (N, D) numpy array
        - chunk_ids: List of chunk IDs aligned with matrix rows
    """
    if not chunks:
        return np.array([]).reshape(0, 0), []

    chunk_ids = [c.chunk_id for c in chunks]
    embeddings = [c.embedding for c in chunks]

    # Convert to float32 for UMAP (it doesn't like float16)
    matrix = np.array(embeddings, dtype=np.float32)

    return matrix, chunk_ids


def compute_similarity_matrix(
    embeddings: np.ndarray,
    threshold: float = 0.8,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, D) embedding matrix.
        threshold: Minimum similarity to keep (others set to 0).

    Returns:
        (N, N) similarity matrix with values below threshold set to 0.
    """
    if embeddings.size == 0:
        return np.array([]).reshape(0, 0)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms

    # Compute cosine similarity
    similarity = normalized @ normalized.T

    # Apply threshold
    similarity = np.where(similarity >= threshold, similarity, 0)

    # Zero out diagonal (no self-similarity)
    np.fill_diagonal(similarity, 0)

    return similarity


__all__ = [
    "fetch_all_chunk_ids",
    "fetch_chunk_embeddings",
    "compress_to_float16",
    "embeddings_to_matrix",
    "compute_similarity_matrix",
]
