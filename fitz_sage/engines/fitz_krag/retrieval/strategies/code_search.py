# fitz_sage/engines/fitz_krag/retrieval/strategies/code_search.py
"""
Symbol-index search: keyword + BM25 + keyword-enrichment boosts.

fitz-sage uses no dense embeddings on symbols. Code retrieval relies
on tree-sitter-extracted symbol names (qualified-name keyword match)
plus full-text BM25 over symbol summaries; precision comes from the
ONNX cross-encoder reranker (``OnnxReranker``) downstream.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.retrieval.strategies.boosts import (
    apply_keyword_enrichment_boost,
    apply_recency_boost,
)
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore

logger = logging.getLogger(__name__)

_SYMBOL_QUERY_STOPWORDS = {
    "and",
    "are",
    "can",
    "does",
    "ever",
    "for",
    "from",
    "how",
    "into",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


class CodeSearchStrategy:
    """Keyword + BM25 search on the symbol index."""

    def __init__(
        self,
        symbol_store: "SymbolStore",
        config: "FitzKragConfig",
    ):
        self._symbol_store = symbol_store
        self._config = config
        self._raw_store: Any = None  # Set by engine for freshness boosting

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
    ) -> list[Address]:
        """
        Retrieve code symbol addresses matching the query.

        1. Keyword search: query words against symbol names
        2. BM25 full-text search (when content_tsv exists)
        3. Merge with configurable keyword-vs-BM25 weights
        4. Keyword-enrichment + freshness boosts
        """
        fetch_limit = limit * 2

        # 1. Keyword search
        keyword_results = self._keyword_results(query, fetch_limit)

        # 2. BM25 search
        bm25_results: list[dict[str, Any]] = []
        try:
            bm25_results = self._symbol_store.search_bm25(query, limit=fetch_limit)
        except Exception as e:
            logger.debug(f"BM25 search not available: {e}")

        # 3. Merge keyword + BM25
        merged = self._merge_results(keyword_results, bm25_results)

        # 4. Keyword enrichment boost (from stored keywords, domain-scaled)
        merged = apply_keyword_enrichment_boost(query, merged, self._symbol_store, detection)

        # 5. Freshness boost (when detection signals boost_recency)
        if detection and getattr(detection, "boost_recency", False) and self._raw_store:
            merged = apply_recency_boost(merged, self._raw_store)

        # 6. Convert to Address objects
        return [self._to_address(r) for r in merged[:limit]]

    def _keyword_results(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search symbol names with deterministic code-identifier query variants."""
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name_query in _symbol_name_queries(query):
            for result in self._symbol_store.search_by_name(name_query, limit=limit):
                sid = str(result["id"])
                if sid in seen:
                    continue
                seen.add(sid)
                results.append(result)
        return results

    def _merge_results(
        self,
        keyword_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Merge keyword + BM25 results with weighted scoring."""
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}
        kw = self._config.keyword_weight
        bw = self._config.code_bm25_weight

        # Normalize across the two legs (we no longer have a semantic leg).
        if bm25_results:
            total = kw + bw
            if total > 0:
                kw, bw = kw / total, bw / total

        # Score keyword results by rank position
        for rank, r in enumerate(keyword_results):
            sid = r["id"]
            rank_score = 1.0 / (rank + 1)
            scores[sid] = scores.get(sid, 0) + kw * rank_score
            by_id[sid] = r

        # Score BM25 results
        if bm25_results:
            for rank, r in enumerate(bm25_results):
                sid = r["id"]
                bm25_score = r.get("bm25_score", 1.0 / (rank + 1))
                scores[sid] = scores.get(sid, 0) + bw * bm25_score
                by_id[sid] = r

        # Sort by combined score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for sid in sorted_ids:
            entry = by_id[sid].copy()
            entry["combined_score"] = scores[sid]
            result.append(entry)
        return result

    def _to_address(self, r: dict[str, Any]) -> Address:
        """Convert a symbol store row to an Address."""
        return Address(
            kind=AddressKind.SYMBOL,
            source_id=r["raw_file_id"],
            location=r["qualified_name"],
            summary=f"{r['kind']} {r['name']}",
            score=r.get("combined_score", 0.0),
            metadata={
                "symbol_id": r["id"],
                "name": r["name"],
                "qualified_name": r["qualified_name"],
                "kind": r["kind"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "signature": r.get("signature"),
            },
        )


def _symbol_name_queries(query: str) -> list[str]:
    """Return bounded literal name-search queries for symbol matching."""
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            variants.append(normalized)

    add(query)
    normalized_query = _normalize_query(query)
    tokens = [
        token
        for token in normalized_query.split()
        if len(token) >= 3 and token not in _SYMBOL_QUERY_STOPWORDS
    ]
    for token in tokens:
        if "_" in token:
            add(token)
    for first, second in zip(tokens, tokens[1:], strict=False):
        add(f"{first}_{second}")
    for token in tokens:
        add(token)
    return variants[:16]


def _normalize_query(query: str) -> str:
    """Normalize prose for deterministic symbol-name query derivation."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_]+", " ", query.lower())).strip()
