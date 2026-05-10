# fitz_sage/engines/fitz_krag/retrieval/strategies/table_search.py
"""
Table search strategy — keyword + semantic hybrid for table metadata.

Semantic is weighted higher than keyword (0.6 vs 0.4) because table
schema summaries provide rich semantic descriptions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore
    from fitz_sage.llm.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class TableSearchStrategy:
    """Hybrid keyword + semantic retrieval for table metadata."""

    def __init__(
        self,
        table_store: "TableStore",
        embedder: "EmbeddingProvider | None",
        config: "FitzKragConfig",
    ):
        self._table_store = table_store
        # ``None`` is the chat-only mode: skip semantic search entirely.
        self._embedder = embedder
        self._config = config

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
        *,
        query_vector: list[float] | None = None,
    ) -> list[Address]:
        """
        Retrieve table addresses matching the query.

        1. Keyword search on table name and column names
        2. Semantic search on schema summaries
        3. Hybrid merge with configurable weights

        Args:
            query_vector: Pre-computed query embedding (skips internal embed call).
        """
        fetch_limit = limit * 2

        # 1. Keyword search
        keyword_results = self._table_store.search_by_name(query, limit=fetch_limit)

        # 2. Semantic search — skipped entirely in chat-only mode.
        semantic_results: list[dict[str, Any]] = []
        if self._embedder is not None:
            try:
                if query_vector is None:
                    query_vector = self._embedder.embed(query, task_type="query")
                semantic_results = self._table_store.search_by_vector(
                    query_vector, limit=fetch_limit
                )
            except Exception as e:
                logger.warning(f"Semantic table search failed, using keyword only: {e}")
                semantic_results = []

        # 3. Hybrid merge
        keyword_weight = self._config.table_keyword_weight
        semantic_weight = self._config.table_semantic_weight
        merged = self._merge_results(
            keyword_results, semantic_results, keyword_weight, semantic_weight
        )

        # 4. Convert to Address objects
        return [self._to_address(r) for r in merged[:limit]]

    def _merge_results(
        self,
        keyword_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
        keyword_weight: float,
        semantic_weight: float,
    ) -> list[dict[str, Any]]:
        """Merge keyword and semantic results with weighted scoring."""
        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        # Normalize keyword scores by rank
        for rank, r in enumerate(keyword_results):
            rid = r["id"]
            rank_score = 1.0 / (rank + 1)
            scores[rid] = scores.get(rid, 0) + keyword_weight * rank_score
            by_id[rid] = r

        # Use cosine scores directly for semantic
        for rank, r in enumerate(semantic_results):
            rid = r["id"]
            cosine_score = r.get("score", 1.0 / (rank + 1))
            scores[rid] = scores.get(rid, 0) + semantic_weight * cosine_score
            by_id[rid] = r

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        result = []
        for rid in sorted_ids:
            entry = by_id[rid].copy()
            entry["combined_score"] = scores[rid]
            result.append(entry)
        return result

    def _to_address(self, record: dict[str, Any]) -> Address:
        """Convert a table store row to an Address."""
        return Address(
            kind=AddressKind.TABLE,
            source_id=record["raw_file_id"],
            location=record["name"],
            summary=record.get("summary") or record["name"],
            score=record.get("combined_score", 0.0),
            metadata={
                "table_index_id": record["id"],
                "table_id": record["table_id"],
                "name": record["name"],
                "columns": record["columns"],
                "row_count": record["row_count"],
            },
        )
