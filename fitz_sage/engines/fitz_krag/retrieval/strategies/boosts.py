# fitz_sage/engines/fitz_krag/retrieval/strategies/boosts.py
"""Shared reciprocal-rank scoring for retrieval strategies."""

from __future__ import annotations

from typing import Any

_RRF_K = 60


def rrf_score(results: list[dict[str, Any]], k: int = _RRF_K) -> list[dict[str, Any]]:
    """Reciprocal-Rank-Fusion score results so combined_score is on a stable scale."""
    scored: list[dict[str, Any]] = []
    for rank, r in enumerate(results):
        entry = r.copy()
        entry["combined_score"] = 1.0 / (k + rank)
        scored.append(entry)
    return scored
