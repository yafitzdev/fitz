# fitz_sage/engines/fitz_krag/retrieval/strategies/code_search.py
"""
Symbol-index search: keyword + BM25 + keyword-enrichment boosts.

fitz-sage uses no dense embeddings on symbols. Code retrieval relies
on tree-sitter-extracted symbol names (qualified-name keyword match)
plus full-text BM25 over symbol summaries; precision comes from the
``LLMReranker`` downstream.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore

logger = logging.getLogger(__name__)


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
        keyword_results = self._symbol_store.search_by_name(query, limit=fetch_limit)

        # 2. BM25 search
        bm25_results: list[dict[str, Any]] = []
        try:
            bm25_results = self._symbol_store.search_bm25(query, limit=fetch_limit)
        except Exception as e:
            logger.debug(f"BM25 search not available: {e}")

        # 3. Merge keyword + BM25
        merged = self._merge_results(keyword_results, bm25_results)

        # 4. Keyword enrichment boost (from stored keywords, domain-scaled)
        merged = self._apply_keyword_enrichment_boost(query, merged, detection)

        # 5. Freshness boost (when detection signals boost_recency)
        if detection and getattr(detection, "boost_recency", False) and self._raw_store:
            merged = self._apply_recency_boost(merged)

        # 6. Convert to Address objects
        return [self._to_address(r) for r in merged[:limit]]

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

    def _apply_recency_boost(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Boost results from recently updated files."""
        if not results:
            return results
        file_ids = list({r["raw_file_id"] for r in results})
        try:
            timestamps = self._raw_store.get_updated_timestamps(file_ids)
            if not timestamps:
                return results
            sorted_files = sorted(timestamps, key=lambda fid: timestamps[fid] or "", reverse=True)
            top_quarter = set(sorted_files[: max(1, len(sorted_files) // 4)])
            top_half = set(sorted_files[: max(1, len(sorted_files) // 2)])
            for r in results:
                fid = r["raw_file_id"]
                if fid in top_quarter:
                    r["combined_score"] = r.get("combined_score", 0) + 0.1
                elif fid in top_half:
                    r["combined_score"] = r.get("combined_score", 0) + 0.05
            results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        except Exception as e:
            logger.debug(f"Recency boost skipped: {e}")
        return results

    def _apply_keyword_enrichment_boost(
        self, query: str, results: list[dict[str, Any]], detection: Any = None
    ) -> list[dict[str, Any]]:
        """Boost results that have matching enriched keywords.

        Domain-specific queries get a stronger keyword boost since terminology
        matches are more meaningful in specialized fields.
        """
        query_terms = [w.lower().strip("?.,!;:()") for w in query.split() if len(w) >= 3]
        if not query_terms:
            return results
        domain = getattr(detection, "domain", "general")
        boost = {"technical": 0.15, "legal": 0.12, "medical": 0.12, "financial": 0.12}.get(
            domain, 0.1
        )
        try:
            keyword_hits = self._symbol_store.search_by_keywords(query_terms, limit=50)
            keyword_ids = {r["id"] for r in keyword_hits}
            if keyword_ids:
                for r in results:
                    if r["id"] in keyword_ids:
                        r["combined_score"] = r.get("combined_score", 0) + boost
                results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        except Exception as e:
            logger.debug(f"Keyword enrichment boost skipped: {e}")
        return results

    def _to_address(self, r: dict[str, Any]) -> Address:
        """Convert a symbol store row to an Address."""
        return Address(
            kind=AddressKind.SYMBOL,
            source_id=r["raw_file_id"],
            location=r["qualified_name"],
            summary=r.get("summary") or f"{r['kind']} {r['name']}",
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
