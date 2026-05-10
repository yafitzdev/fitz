# fitz_sage/engines/fitz_krag/retrieval/strategies/section_search.py
"""
Section search strategy — BM25 + semantic hybrid for technical documents.

BM25 is weighted higher than semantic (0.6 vs 0.4) because technical
documents are keyword-heavy and full-text search excels at exact term matching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore
    from fitz_sage.llm.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class SectionSearchStrategy:
    """BM25-first retrieval with semantic fallback for technical documents."""

    def __init__(
        self,
        section_store: "SectionStore",
        embedder: "EmbeddingProvider | None",
        config: "FitzKragConfig",
    ):
        self._section_store = section_store
        # ``None`` is the chat-only mode: skip semantic search entirely.
        self._embedder = embedder
        self._config = config
        self._raw_store: Any = None  # Set by engine for freshness boosting

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
        *,
        query_vector: list[float] | None = None,
        hyde_vectors: list[list[float]] | None = None,
        inject_corpus_summaries: bool = False,
    ) -> list[Address]:
        """
        Retrieve section addresses matching the query.

        1. BM25 full-text search (PostgreSQL ts_rank)
        2. Semantic search on section summaries
        3. HyDE search (when enabled)
        4. Hybrid merge — BM25 weighted higher

        Args:
            query_vector: Pre-computed query embedding (skips internal embed call).
            hyde_vectors: Pre-computed HyDE hypothesis embeddings (skips HyDE generate + embed).
            inject_corpus_summaries: When True, skip normal search and return L2 corpus
                summary chunks only. Used by the router for thematic query enrichment.
        """
        if inject_corpus_summaries:
            return [self._to_address(s) for s in self._section_store.get_corpus_summaries()]
        fetch_limit = limit * 2

        # 1. BM25 search
        bm25_results = self._section_store.search_bm25(query, limit=fetch_limit)

        # 2. Semantic search — skipped entirely in chat-only mode.
        semantic_results: list[dict[str, Any]] = []
        if self._embedder is not None:
            try:
                if query_vector is None:
                    query_vector = self._embedder.embed(query, task_type="query")
                semantic_results = self._section_store.search_by_vector(
                    query_vector, limit=fetch_limit
                )
            except Exception as e:
                logger.warning(f"Semantic section search failed, using BM25 only: {e}")
                semantic_results = []

        # 3. HyDE search (uses pre-computed vectors from router)
        if hyde_vectors:
            hyde_results = self._run_hyde(query, fetch_limit, hyde_vectors=hyde_vectors)
            semantic_results = self._merge_hyde(semantic_results, hyde_results)

        # 4. Hybrid merge
        bm25_weight = self._config.section_bm25_weight
        semantic_weight = self._config.section_semantic_weight
        merged = self._merge_results(bm25_results, semantic_results, bm25_weight, semantic_weight)

        # 5. Keyword enrichment boost (from stored keywords, domain-scaled)
        merged = self._apply_keyword_enrichment_boost(query, merged, detection)

        # 6. Freshness boost (when detection signals boost_recency)
        if detection and getattr(detection, "boost_recency", False) and self._raw_store:
            merged = self._apply_recency_boost(merged)

        # 7. Enrich with parent titles for breadcrumb location
        top_results = merged[:limit]
        self._enrich_with_parent_titles(top_results)

        # 8. Convert to Address objects
        return [self._to_address(r) for r in top_results]

    def _run_hyde(
        self,
        query: str,
        limit: int,
        *,
        hyde_vectors: list[list[float]],
    ) -> list[dict[str, Any]]:
        """Search with pre-computed HyDE vectors from the router."""
        try:
            all_results: list[dict[str, Any]] = []
            for vec in hyde_vectors:
                results = self._section_store.search_by_vector(vec, limit=limit)
                all_results.extend(results)
            return all_results
        except Exception as e:
            logger.warning(f"HyDE section search failed: {e}")
            return []

    def _merge_hyde(
        self,
        semantic_results: list[dict[str, Any]],
        hyde_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge HyDE results into semantic results with lower weight."""
        for r in hyde_results:
            r["score"] = r.get("score", 0.0) * 0.5
        combined = list(semantic_results)
        seen_ids = {r["id"] for r in combined}
        for r in hyde_results:
            if r["id"] not in seen_ids:
                combined.append(r)
                seen_ids.add(r["id"])
        return combined

    def _merge_results(
        self,
        bm25_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
        bm25_weight: float,
        semantic_weight: float,
    ) -> list[dict[str, Any]]:
        """Merge BM25 and semantic results using Reciprocal Rank Fusion (RRF).

        Both legs use RRF with k=60 so they are on the same scale before
        weighting. This avoids the asymmetry of mixing raw cosine scores with
        rank-reciprocal BM25 scores.
        """
        _RRF_K = 60
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        for rank, r in enumerate(bm25_results):
            sid = r["id"]
            rrf_score = 1.0 / (_RRF_K + rank)
            scores[sid] = scores.get(sid, 0) + bm25_weight * rrf_score
            by_id[sid] = r

        for rank, r in enumerate(semantic_results):
            sid = r["id"]
            rrf_score = 1.0 / (_RRF_K + rank)
            scores[sid] = scores.get(sid, 0) + semantic_weight * rrf_score
            by_id[sid] = r

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
        # Domain signal: amplify keyword importance for specialized domains
        domain = getattr(detection, "domain", "general")
        boost = {"technical": 0.15, "legal": 0.12, "medical": 0.12, "financial": 0.12}.get(
            domain, 0.1
        )
        try:
            keyword_hits = self._section_store.search_by_keywords(query_terms, limit=50)
            keyword_ids = {r["id"] for r in keyword_hits}
            if keyword_ids:
                for r in results:
                    if r["id"] in keyword_ids:
                        r["combined_score"] = r.get("combined_score", 0) + boost
                # Re-sort after boost
                results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        except Exception as e:
            logger.debug(f"Keyword enrichment boost skipped: {e}")
        return results

    def _enrich_with_parent_titles(self, results: list[dict[str, Any]]) -> None:
        """Batch-fetch parent section titles and attach to results.

        This enables breadcrumb-style location (e.g. "Model X100 > Specifications")
        so that the ranker's entity match bonus considers parent context.
        """
        parent_ids = {r["parent_section_id"] for r in results if r.get("parent_section_id")}
        if not parent_ids:
            return

        parent_titles: dict[str, str] = {}
        for pid in parent_ids:
            parent = self._section_store.get(pid)
            if parent:
                parent_titles[pid] = parent["title"]

        for r in results:
            pid = r.get("parent_section_id")
            if pid and pid in parent_titles:
                r["parent_title"] = parent_titles[pid]

    def _to_address(self, section: dict[str, Any]) -> Address:
        """Convert a section store row to an Address."""
        # Build breadcrumb location from parent title when available
        title = section["title"]
        parent_title = section.get("parent_title")
        location = f"{parent_title} > {title}" if parent_title else title

        return Address(
            kind=AddressKind.SECTION,
            source_id=section["raw_file_id"],
            location=location,
            summary=section.get("summary") or section["title"],
            score=section.get("combined_score", 0.0),
            metadata={
                "section_id": section["id"],
                "level": section["level"],
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "parent_section_id": section.get("parent_section_id"),
            },
        )
