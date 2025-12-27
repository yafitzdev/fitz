# fitz_ai/map/clustering.py
"""
Cluster detection for knowledge map using K-means.

Groups related chunks together for visualization.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from fitz_ai.map.models import ChunkEmbedding, ClusterInfo

logger = logging.getLogger(__name__)

# Cluster colors (GitHub-style palette)
CLUSTER_COLORS = [
    "#58a6ff",  # Blue
    "#a371f7",  # Purple
    "#3fb950",  # Green
    "#f0883e",  # Orange
    "#f85149",  # Red
    "#db61a2",  # Pink
    "#79c0ff",  # Light blue
    "#7ee787",  # Light green
    "#ffa657",  # Light orange
    "#ff7b72",  # Light red
]


def detect_clusters(
    coordinates: np.ndarray,
    chunk_ids: List[str],
    n_clusters: int | None = None,
    max_clusters: int = 10,
    min_cluster_size: int = 3,
) -> Tuple[np.ndarray, List[ClusterInfo]]:
    """
    Detect clusters in 2D projected space using K-means.

    Args:
        coordinates: (N, 2) UMAP coordinates.
        chunk_ids: Aligned chunk IDs.
        n_clusters: Number of clusters. If None, auto-detect using elbow method.
        max_clusters: Maximum clusters to consider for auto-detection.
        min_cluster_size: Minimum chunks for a valid cluster.

    Returns:
        Tuple of (cluster_labels, cluster_info_list)
        cluster_labels is array of length N with cluster IDs.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for knowledge map. "
            "Install with: pip install fitz-ai[map]"
        )

    n_samples = len(chunk_ids)

    if n_samples == 0:
        return np.array([]), []

    if n_samples < min_cluster_size:
        # Too few samples for meaningful clustering
        labels = np.zeros(n_samples, dtype=int)
        cluster_info = [
            ClusterInfo(
                cluster_id=0,
                label="All",
                chunk_count=n_samples,
                centroid_x=float(coordinates[:, 0].mean()) if n_samples > 0 else 0,
                centroid_y=float(coordinates[:, 1].mean()) if n_samples > 0 else 0,
                color=CLUSTER_COLORS[0],
            )
        ]
        return labels, cluster_info

    # Auto-detect optimal k if not specified
    if n_clusters is None:
        n_clusters = optimal_k(coordinates, max_clusters=max_clusters)

    # Ensure we don't have more clusters than samples
    n_clusters = min(n_clusters, n_samples)

    logger.info(f"Running K-means with {n_clusters} clusters on {n_samples} points")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    labels = kmeans.fit_predict(coordinates)

    # Build cluster info
    cluster_info = []
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        count = int(mask.sum())

        if count == 0:
            continue

        cluster_coords = coordinates[mask]
        centroid_x = float(cluster_coords[:, 0].mean())
        centroid_y = float(cluster_coords[:, 1].mean())

        cluster_info.append(
            ClusterInfo(
                cluster_id=cluster_id,
                label=f"Cluster {cluster_id + 1}",
                chunk_count=count,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            )
        )

    logger.info(f"Found {len(cluster_info)} clusters")
    return labels, cluster_info


def optimal_k(
    coordinates: np.ndarray,
    max_clusters: int = 10,
    min_clusters: int = 2,
) -> int:
    """
    Find optimal number of clusters using elbow method.

    Uses the point of maximum curvature in the inertia curve.

    Args:
        coordinates: (N, 2) coordinate matrix.
        max_clusters: Maximum k to consider.
        min_clusters: Minimum k to consider.

    Returns:
        Optimal number of clusters.
    """
    from sklearn.cluster import KMeans

    n_samples = len(coordinates)
    if n_samples <= min_clusters:
        return min(min_clusters, n_samples)

    max_k = min(max_clusters, n_samples - 1)
    if max_k < min_clusters:
        return min_clusters

    inertias = []
    k_range = range(min_clusters, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(coordinates)
        inertias.append(kmeans.inertia_)

    if len(inertias) < 3:
        return min_clusters

    # Find elbow using second derivative (point of maximum curvature)
    # Simple approach: find where the rate of decrease slows most
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)

    # The elbow is where the second derivative is maximum (least negative)
    if len(second_diffs) > 0:
        elbow_idx = np.argmax(second_diffs) + min_clusters
    else:
        elbow_idx = min_clusters

    # Ensure we return a reasonable value
    optimal = max(min_clusters, min(elbow_idx, max_k))

    logger.debug(f"Optimal k determined: {optimal}")
    return optimal


def assign_cluster_labels(
    chunks: List[ChunkEmbedding],
    cluster_labels: np.ndarray,
    chunk_id_order: List[str],
) -> List[ChunkEmbedding]:
    """
    Assign cluster IDs to chunks.

    Args:
        chunks: ChunkEmbedding objects to update.
        cluster_labels: Array of cluster IDs.
        chunk_id_order: Order of chunk_ids in labels array.

    Returns:
        Updated chunks with cluster_id populated.
    """
    # Build lookup from chunk_id to cluster label
    label_lookup = {}
    for i, chunk_id in enumerate(chunk_id_order):
        if i < len(cluster_labels):
            label_lookup[chunk_id] = int(cluster_labels[i])

    # Assign to chunks
    for chunk in chunks:
        if chunk.chunk_id in label_lookup:
            chunk.cluster_id = label_lookup[chunk.chunk_id]

    return chunks


def extract_cluster_keywords(
    chunks: List[ChunkEmbedding],
    cluster_id: int,
    top_k: int = 5,
) -> List[str]:
    """
    Extract top keywords for a cluster using simple word frequency.

    Args:
        chunks: All chunks (will filter to cluster).
        cluster_id: Cluster to analyze.
        top_k: Number of keywords to return.

    Returns:
        List of top keywords.
    """
    import re
    from collections import Counter

    # Filter to cluster
    cluster_chunks = [c for c in chunks if c.cluster_id == cluster_id]

    if not cluster_chunks:
        return []

    # Simple word extraction (could use TF-IDF for better results)
    words = []
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "now",
        "here",
        "there",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "i",
        "my",
    }

    for chunk in cluster_chunks:
        text = chunk.content_preview.lower()
        # Extract words (alphanumeric, 3+ chars)
        chunk_words = re.findall(r"\b[a-z][a-z0-9]{2,}\b", text)
        words.extend(w for w in chunk_words if w not in stop_words)

    # Get most common
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_k)]


__all__ = [
    "detect_clusters",
    "optimal_k",
    "assign_cluster_labels",
    "extract_cluster_keywords",
    "CLUSTER_COLORS",
]
