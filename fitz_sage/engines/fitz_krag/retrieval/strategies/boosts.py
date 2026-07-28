# fitz_sage/engines/fitz_krag/retrieval/strategies/boosts.py
"""Shared reciprocal-rank and recency scoring for retrieval strategies."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_RRF_K = 60

def rrf_score(results: list[dict[str, Any]], k: int = _RRF_K) -> list[dict[str, Any]]:
    """Reciprocal-Rank-Fusion score results so combined_score is on a stable scale."""
    scored: list[dict[str, Any]] = []
    for rank, r in enumerate(results):
        entry = r.copy()
        entry["combined_score"] = 1.0 / (k + rank)
        scored.append(entry)
    return scored


def apply_recency_boost(results: list[dict[str, Any]], raw_store: Any) -> list[dict[str, Any]]:
    """Multiplicatively boost results from recently updated files."""
    if not results:
        return results
    file_ids = list({r["raw_file_id"] for r in results})
    try:
        timestamps = raw_store.get_updated_timestamps(file_ids)
        if not timestamps:
            return results
        sorted_files = sorted(timestamps, key=lambda fid: timestamps[fid] or "", reverse=True)
        top_quarter = set(sorted_files[: max(1, len(sorted_files) // 4)])
        top_half = set(sorted_files[: max(1, len(sorted_files) // 2)])
        for r in results:
            fid = r["raw_file_id"]
            # Multiplicative boost — proportionate to each strategy's score scale.
            if fid in top_quarter:
                r["combined_score"] = r.get("combined_score", 0) * 1.5
            elif fid in top_half:
                r["combined_score"] = r.get("combined_score", 0) * 1.2
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    except Exception as e:
        logger.debug(f"Recency boost skipped: {e}")
    return results
