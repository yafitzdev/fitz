# fitz_sage/engines/fitz_krag/retrieval/strategies/boosts.py
"""Shared scoring for index-search retrieval strategies.

``CodeSearchStrategy`` (symbol index) and ``SectionSearchStrategy`` (section
index) apply the same RRF scaling, keyword-enrichment, and recency boosts;
``TableSearchStrategy`` shares the RRF scaling. These functions are the single
implementation — each strategy passes its own store.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_RRF_K = 60

# Domain → keyword-enrichment boost. Specialized terminology matches are more
# meaningful, so domain-specific queries get a stronger boost.
_DOMAIN_BOOST = {"technical": 0.15, "legal": 0.12, "medical": 0.12, "financial": 0.12}
_DEFAULT_BOOST = 0.1


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


def apply_keyword_enrichment_boost(
    query: str,
    results: list[dict[str, Any]],
    store: Any,
    detection: Any = None,
) -> list[dict[str, Any]]:
    """Boost results whose stored enriched keywords match the query (domain-scaled).

    ``store`` must expose ``search_by_keywords(terms, limit)`` returning rows with
    an ``id`` field (SymbolStore or SectionStore).
    """
    query_terms = [w.lower().strip("?.,!;:()") for w in query.split() if len(w) >= 3]
    if not query_terms:
        return results
    domain = getattr(detection, "domain", "general")
    boost = _DOMAIN_BOOST.get(domain, _DEFAULT_BOOST)
    try:
        keyword_hits = store.search_by_keywords(query_terms, limit=50)
        keyword_ids = {r["id"] for r in keyword_hits}
        if keyword_ids:
            for r in results:
                if r["id"] in keyword_ids:
                    r["combined_score"] = r.get("combined_score", 0) + boost
            results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
    except Exception as e:
        logger.debug(f"Keyword enrichment boost skipped: {e}")
    return results
