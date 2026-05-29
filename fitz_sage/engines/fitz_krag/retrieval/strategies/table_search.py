# fitz_sage/engines/fitz_krag/retrieval/strategies/table_search.py
"""
Table search strategy — keyword retrieval over table-name + column-name index.

fitz-sage uses no dense embeddings. Tables are surfaced by name/column
keyword match; precision comes from the ONNX cross-encoder reranker
(``OnnxReranker``) downstream (or, for actual data answers, from
``TableQueryHandler`` which runs SQL against the matched table).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.retrieval.strategies.boosts import rrf_score
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore

logger = logging.getLogger(__name__)


class TableSearchStrategy:
    """Keyword retrieval for table metadata."""

    def __init__(
        self,
        table_store: "TableStore",
        config: "FitzKragConfig",
    ):
        self._table_store = table_store
        self._config = config

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
    ) -> list[Address]:
        """Retrieve table addresses matching the query by name/column keyword match."""
        fetch_limit = limit * 2

        keyword_results = self._table_store.search_by_name(query, limit=fetch_limit)

        # RRF-score the single retrieval leg for consistent combined_score scaling.
        scored = rrf_score(keyword_results[:limit])

        return [self._to_address(r) for r in scored]

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
