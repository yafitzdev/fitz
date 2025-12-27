# fitz_ai/map/projection.py
"""
UMAP projection for dimensionality reduction.

Reduces high-dimensional embeddings to 2D for visualization.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from fitz_ai.map.models import ChunkEmbedding, DocumentNode

logger = logging.getLogger(__name__)


def run_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction on embeddings.

    Args:
        embeddings: (N, D) matrix of embeddings.
        n_neighbors: UMAP n_neighbors parameter (local neighborhood size).
        min_dist: UMAP min_dist parameter (minimum distance between points).
        metric: Distance metric (cosine works well for embeddings).
        random_state: For reproducibility.

    Returns:
        (N, 2) matrix of 2D coordinates.

    Raises:
        ImportError: If umap-learn is not installed.
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for knowledge map. "
            "Install with: pip install fitz-ai[map]"
        )

    if embeddings.size == 0:
        return np.array([]).reshape(0, 2)

    n_samples = embeddings.shape[0]

    # Handle edge cases
    if n_samples == 1:
        return np.array([[0.0, 0.0]])

    if n_samples == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]])

    # Adjust n_neighbors if we have fewer samples
    effective_neighbors = min(n_neighbors, n_samples - 1)

    logger.info(
        f"Running UMAP on {n_samples} embeddings (n_neighbors={effective_neighbors})"
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )

    coordinates = reducer.fit_transform(embeddings)

    # Normalize to [-1, 1] range for consistent visualization
    coordinates = normalize_coordinates(coordinates)

    logger.info("UMAP projection complete")
    return coordinates


def normalize_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """
    Normalize 2D coordinates to [-1, 1] range.

    Args:
        coordinates: (N, 2) array of coordinates.

    Returns:
        Normalized (N, 2) array.
    """
    if coordinates.size == 0:
        return coordinates

    # Get min/max for each dimension
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    ranges = maxs - mins

    # Avoid division by zero
    ranges = np.where(ranges == 0, 1, ranges)

    # Normalize to [0, 1] then scale to [-1, 1]
    normalized = (coordinates - mins) / ranges
    normalized = normalized * 2 - 1

    return normalized


def assign_coordinates(
    chunks: List[ChunkEmbedding],
    coordinates: np.ndarray,
    chunk_id_order: List[str],
) -> List[ChunkEmbedding]:
    """
    Assign UMAP coordinates to ChunkEmbedding objects.

    Args:
        chunks: ChunkEmbedding objects to update.
        coordinates: (N, 2) UMAP output.
        chunk_id_order: Order of chunk_ids in coordinates matrix.

    Returns:
        Updated ChunkEmbedding objects with x, y populated.
    """
    # Build lookup from chunk_id to coordinates
    coord_lookup: Dict[str, tuple] = {}
    for i, chunk_id in enumerate(chunk_id_order):
        if i < len(coordinates):
            coord_lookup[chunk_id] = (
                float(coordinates[i, 0]),
                float(coordinates[i, 1]),
            )

    # Assign coordinates to chunks
    for chunk in chunks:
        if chunk.chunk_id in coord_lookup:
            chunk.x, chunk.y = coord_lookup[chunk.chunk_id]

    return chunks


def compute_document_centroids(
    documents: Dict[str, DocumentNode],
    chunks: List[ChunkEmbedding],
) -> Dict[str, DocumentNode]:
    """
    Compute document positions as centroids of their child chunks.

    Args:
        documents: Document nodes keyed by doc_id.
        chunks: Chunks with coordinates assigned.

    Returns:
        Updated documents with x, y set to centroid of children.
    """
    # Build chunk lookup
    chunk_lookup = {c.chunk_id: c for c in chunks}

    for doc_id, doc in documents.items():
        if not doc.chunk_ids:
            continue

        # Collect child coordinates
        xs = []
        ys = []
        for chunk_id in doc.chunk_ids:
            chunk = chunk_lookup.get(chunk_id)
            if chunk and chunk.x is not None and chunk.y is not None:
                xs.append(chunk.x)
                ys.append(chunk.y)

        # Compute centroid
        if xs and ys:
            doc.x = sum(xs) / len(xs)
            doc.y = sum(ys) / len(ys)

    return documents


__all__ = [
    "run_umap_projection",
    "normalize_coordinates",
    "assign_coordinates",
    "compute_document_centroids",
]
