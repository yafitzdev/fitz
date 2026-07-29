# fitz_sage/engines/fitz_krag/retrieval/strategies/section_search.py
"""
Section search strategy — BM25 retrieval for technical documents.

fitz-sage uses no dense embeddings. Section retrieval is SQLite FTS5
``bm25()``; precision comes from the ONNX cross-encoder reranker
(``OnnxReranker``) downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.retrieval.snippets import query_relevant_excerpt
from fitz_sage.engines.fitz_krag.retrieval.strategies.boosts import rrf_score
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore

_RERANK_TEXT_CHARS = 1200


class SectionSearchStrategy:
    """BM25 retrieval for technical documents."""

    def __init__(
        self,
        section_store: "SectionStore",
        raw_store: Any,
        config: "FitzKragConfig",
    ):
        self._section_store = section_store
        self._raw_store = raw_store
        self._config = config

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
        *,
        inject_corpus_summaries: bool = False,
    ) -> list[Address]:
        """
        Retrieve section addresses matching the query.

        1. BM25 full-text search (SQLite FTS5 bm25())
        2. Parent-title breadcrumb enrichment
        """
        if inject_corpus_summaries:
            return [
                self._to_address(s, query=query) for s in self._section_store.get_corpus_summaries()
            ]
        fetch_limit = limit * 2

        # 1. BM25 search — the canonical retrieval signal.
        bm25_results = self._section_store.search_bm25(query, limit=fetch_limit)

        # 2. Score BM25 hits via Reciprocal Rank Fusion for stable scaling.
        merged = rrf_score(bm25_results)

        # 3. Enrich with parent titles for breadcrumb location
        top_results = merged[:limit]
        self._enrich_with_parent_titles(top_results)

        # 4. Convert to Address objects
        return [self._to_address(r, query=query) for r in top_results]

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

    def _to_address(self, section: dict[str, Any], *, query: str = "") -> Address:
        """Convert a section store row to an Address."""
        # Build breadcrumb location from parent title when available
        title = section["title"]
        parent_title = section.get("parent_title")
        location = f"{parent_title} > {title}" if parent_title else title

        # Demand-driven cold path: when no LLM summary exists yet, give the
        # cross-encoder reranker a content snippet (not just the title) so it
        # has real text to score against. Once the warm loop summarizes the
        # file, the real summary replaces this.
        summary = section.get("summary")
        has_summary = bool(summary)
        if not has_summary:
            content = (section.get("content") or "").strip()
            summary = f"{title}: {content[:300]}" if content else title

        metadata = dict(section.get("metadata") or {})
        content = (section.get("content") or "").strip()
        if query and len(content) > _RERANK_TEXT_CHARS:
            metadata["rerank_text"] = query_relevant_excerpt(
                query,
                content,
                max_chars=_RERANK_TEXT_CHARS,
            )
            if not has_summary:
                metadata["rerank_heading"] = location
        raw_file = self._raw_store.get(section["raw_file_id"]) if self._raw_store else None
        if raw_file and raw_file.get("path"):
            metadata["source_path"] = raw_file["path"]

        metadata.update(
            {
                "section_id": section["id"],
                "level": section["level"],
                "page_start": section.get("page_start"),
                "page_end": section.get("page_end"),
                "parent_section_id": section.get("parent_section_id"),
            }
        )

        return Address(
            kind=AddressKind.SECTION,
            source_id=section["raw_file_id"],
            location=location,
            summary=summary,
            score=section.get("combined_score", 0.0),
            metadata=metadata,
        )
