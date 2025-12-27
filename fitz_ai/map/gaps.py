# fitz_ai/map/gaps.py
"""
Gap detection for knowledge map.

Identifies sparse regions and isolated chunks in the knowledge base.
"""

from __future__ import annotations

import logging
from typing import List, Set, Tuple

import numpy as np

from fitz_ai.map.models import ChunkEmbedding, ClusterInfo, GapInfo

logger = logging.getLogger(__name__)


def detect_gaps(
    chunks: List[ChunkEmbedding],
    cluster_info: List[ClusterInfo],
    coordinates: np.ndarray,
    chunk_id_order: List[str],
    isolation_percentile: float = 90,
    small_cluster_threshold: int = 3,
) -> Tuple[List[GapInfo], Set[str]]:
    """
    Detect gaps in knowledge coverage.

    Gap detection strategy:
    1. Cluster outliers: Points far from their cluster centroid
    2. Low local density: Points with few neighbors
    3. Small clusters: Clusters with very few chunks

    Args:
        chunks: All chunks with coordinates and cluster_ids.
        cluster_info: Cluster information.
        coordinates: (N, 2) UMAP coordinates.
        chunk_id_order: Order of chunk_ids in coordinates.
        isolation_percentile: Percentile threshold for isolation detection.
        small_cluster_threshold: Clusters smaller than this are considered sparse.

    Returns:
        Tuple of (gap_info_list, set of chunk_ids marked as gaps)
    """
    if not chunks or len(coordinates) == 0:
        return [], set()

    gap_chunk_ids: Set[str] = set()
    gaps: List[GapInfo] = []
    gap_id = 0

    # Build lookups
    id_to_idx = {cid: i for i, cid in enumerate(chunk_id_order)}

    # 1. Detect cluster outliers (far from centroid)
    centroid_lookup = {
        ci.cluster_id: (ci.centroid_x, ci.centroid_y) for ci in cluster_info
    }

    outlier_ids, outlier_coords = _detect_cluster_outliers(
        chunks, coordinates, chunk_id_order, centroid_lookup, isolation_percentile
    )
    gap_chunk_ids.update(outlier_ids)

    if outlier_ids:
        gaps.append(
            GapInfo(
                gap_id=gap_id,
                description="Chunks far from cluster centers (potential outliers)",
                x=float(np.mean([c[0] for c in outlier_coords]))
                if outlier_coords
                else 0,
                y=float(np.mean([c[1] for c in outlier_coords]))
                if outlier_coords
                else 0,
                severity="medium",
                isolated_chunk_ids=list(outlier_ids),
            )
        )
        gap_id += 1

    # 2. Detect low local density points
    isolated_ids, isolated_coords = _detect_low_density(
        coordinates, chunk_id_order, isolation_percentile
    )
    new_isolated = isolated_ids - gap_chunk_ids
    gap_chunk_ids.update(new_isolated)

    if new_isolated:
        gaps.append(
            GapInfo(
                gap_id=gap_id,
                description="Isolated chunks with few neighbors",
                x=float(
                    np.mean([coordinates[id_to_idx[cid], 0] for cid in new_isolated])
                ),
                y=float(
                    np.mean([coordinates[id_to_idx[cid], 1] for cid in new_isolated])
                ),
                severity="high",
                isolated_chunk_ids=list(new_isolated),
            )
        )
        gap_id += 1

    # 3. Detect small clusters
    small_cluster_ids = _detect_small_clusters(
        chunks, cluster_info, small_cluster_threshold
    )
    new_small = small_cluster_ids - gap_chunk_ids
    gap_chunk_ids.update(new_small)

    if new_small:
        gaps.append(
            GapInfo(
                gap_id=gap_id,
                description="Sparse coverage areas (small clusters)",
                x=float(
                    np.mean(
                        [
                            coordinates[id_to_idx[cid], 0]
                            for cid in new_small
                            if cid in id_to_idx
                        ]
                    )
                ),
                y=float(
                    np.mean(
                        [
                            coordinates[id_to_idx[cid], 1]
                            for cid in new_small
                            if cid in id_to_idx
                        ]
                    )
                ),
                severity="low",
                isolated_chunk_ids=list(new_small),
            )
        )
        gap_id += 1

    logger.info(
        f"Detected {len(gaps)} gap regions with {len(gap_chunk_ids)} affected chunks"
    )
    return gaps, gap_chunk_ids


def _detect_cluster_outliers(
    chunks: List[ChunkEmbedding],
    coordinates: np.ndarray,
    chunk_id_order: List[str],
    centroid_lookup: dict,
    percentile: float,
) -> Tuple[Set[str], List[Tuple[float, float]]]:
    """Find chunks that are far from their cluster centroid."""
    if not centroid_lookup:
        return set(), []

    id_to_idx = {cid: i for i, cid in enumerate(chunk_id_order)}
    distances = []

    for chunk in chunks:
        if chunk.cluster_id is None or chunk.chunk_id not in id_to_idx:
            continue

        centroid = centroid_lookup.get(chunk.cluster_id)
        if centroid is None:
            continue

        idx = id_to_idx[chunk.chunk_id]
        x, y = coordinates[idx]
        dist = np.sqrt((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2)
        distances.append((chunk.chunk_id, dist, (x, y)))

    if not distances:
        return set(), []

    # Find threshold at given percentile
    all_dists = [d[1] for d in distances]
    threshold = np.percentile(all_dists, percentile)

    # Return chunks beyond threshold
    outliers = [(cid, coord) for cid, dist, coord in distances if dist > threshold]
    return set(o[0] for o in outliers), [o[1] for o in outliers]


def _detect_low_density(
    coordinates: np.ndarray,
    chunk_id_order: List[str],
    percentile: float,
    k: int = 5,
) -> Tuple[Set[str], List[Tuple[float, float]]]:
    """Find chunks in low-density regions using k-nearest neighbor distance."""
    n = len(coordinates)
    if n < k + 1:
        return set(), []

    # Compute pairwise distances
    # Using simple Euclidean distance in 2D
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
            distances[i, j] = d
            distances[j, i] = d

    # For each point, find k-th nearest neighbor distance
    knn_distances = []
    for i in range(n):
        sorted_dists = np.sort(distances[i])
        # k+1 because first is always 0 (self)
        knn_dist = sorted_dists[min(k, n - 1)]
        knn_distances.append(knn_dist)

    knn_distances = np.array(knn_distances)

    # Points with high knn distance are isolated
    threshold = np.percentile(knn_distances, percentile)
    isolated_indices = np.where(knn_distances > threshold)[0]

    isolated_ids = {chunk_id_order[i] for i in isolated_indices}
    isolated_coords = [(coordinates[i, 0], coordinates[i, 1]) for i in isolated_indices]

    return isolated_ids, isolated_coords


def _detect_small_clusters(
    chunks: List[ChunkEmbedding],
    cluster_info: List[ClusterInfo],
    threshold: int,
) -> Set[str]:
    """Find chunks in clusters smaller than threshold."""
    small_cluster_ids = {
        ci.cluster_id for ci in cluster_info if ci.chunk_count < threshold
    }

    if not small_cluster_ids:
        return set()

    return {
        c.chunk_id
        for c in chunks
        if c.cluster_id is not None and c.cluster_id in small_cluster_ids
    }


def mark_gap_chunks(
    chunks: List[ChunkEmbedding],
    gap_chunk_ids: Set[str],
) -> List[ChunkEmbedding]:
    """Mark chunks that are in gap regions."""
    for chunk in chunks:
        chunk.is_gap = chunk.chunk_id in gap_chunk_ids
    return chunks


def compute_coverage_score(
    total_chunks: int,
    gap_chunk_count: int,
) -> float:
    """
    Compute coverage score (1.0 = perfect coverage, 0.0 = all gaps).

    Args:
        total_chunks: Total number of chunks.
        gap_chunk_count: Number of chunks marked as gaps.

    Returns:
        Coverage score between 0 and 1.
    """
    if total_chunks == 0:
        return 1.0

    return 1.0 - (gap_chunk_count / total_chunks)


__all__ = [
    "detect_gaps",
    "mark_gap_chunks",
    "compute_coverage_score",
]
