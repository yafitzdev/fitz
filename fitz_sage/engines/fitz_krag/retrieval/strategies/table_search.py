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

from fitz_sage.engines.fitz_krag.evidence_contract import exact_identifiers
from fitz_sage.engines.fitz_krag.retrieval.strategies.boosts import rrf_score
from fitz_sage.engines.fitz_krag.retrieval.table_plan import (
    build_table_query_plan,
    execute_table_query_plan,
)
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore
    from fitz_sage.tabular.store.sqlite import SqliteTableStore

logger = logging.getLogger(__name__)
_ROW_SCAN_LIMIT = 500


class TableSearchStrategy:
    """Keyword retrieval for table metadata."""

    def __init__(
        self,
        table_store: "TableStore",
        config: "FitzKragConfig",
        sqlite_table_store: "SqliteTableStore | None" = None,
    ):
        self._table_store = table_store
        self._config = config
        self._sqlite_table_store = sqlite_table_store

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
    ) -> list[Address]:
        """Retrieve table addresses matching the query by name/column keyword match."""
        fetch_limit = limit * 2

        keyword_results = self._table_store.search_by_name(query, limit=fetch_limit)
        row_results = self._row_results(query, limit=fetch_limit, profile=detection)

        # RRF-score the single retrieval leg for consistent combined_score scaling.
        scored = _merge_table_results(rrf_score(keyword_results), row_results)[:limit]

        return [self._to_address(r) for r in scored]

    def _row_results(
        self,
        query: str,
        *,
        limit: int,
        profile: Any = None,
    ) -> list[dict[str, Any]]:
        """Return table-index rows whose concrete data satisfies the query."""
        if not self._should_scan_rows(query, profile):
            return []
        catalog = getattr(self._sqlite_table_store, "catalog", None)
        scan_rows = getattr(self._sqlite_table_store, "scan_rows", None)
        get_by_table_id = getattr(self._table_store, "get_by_table_id", None)
        if not callable(catalog) or not callable(scan_rows) or not callable(get_by_table_id):
            return []

        matches: list[dict[str, Any]] = []
        for table in catalog():
            table_id = str(table.get("table_id") or "")
            if not table_id:
                continue
            scan = scan_rows(table_id, limit=_ROW_SCAN_LIMIT)
            if scan is None:
                continue
            columns, rows = scan
            if not columns or not rows:
                continue
            plan = build_table_query_plan(query, columns, rows)
            selected_rows = execute_table_query_plan(plan, rows)
            if not selected_rows:
                continue
            record = get_by_table_id(table_id)
            if not record:
                continue
            metadata = dict(record.get("metadata") or {})
            metadata["row_match"] = {
                "matched_rows": len(selected_rows),
                "plan": plan.metadata,
            }
            entry = {
                **record,
                "metadata": metadata,
                "combined_score": _row_match_score(plan.metadata, len(selected_rows)),
            }
            matches.append(entry)

        matches.sort(key=lambda item: item.get("combined_score", 0.0), reverse=True)
        return matches[:limit]

    def _should_scan_rows(self, query: str, profile: Any = None) -> bool:
        """Return whether bounded row scanning is warranted for this query."""
        required = tuple(getattr(profile, "required_modalities", ()) or ())
        if "table" in required:
            return True
        return bool(exact_identifiers(query))

    def _to_address(self, record: dict[str, Any]) -> Address:
        """Convert a table store row to an Address."""
        columns = list(record["columns"])
        summary = record.get("summary") or (
            f"Table {record['name']} columns: {', '.join(columns)}. "
            f"Rows: {record['row_count']}."
        )
        metadata = dict(record.get("metadata") or {})
        metadata.update(
            {
                "table_index_id": record["id"],
                "table_id": record["table_id"],
                "name": record["name"],
                "columns": columns,
                "row_count": record["row_count"],
            }
        )
        return Address(
            kind=AddressKind.TABLE,
            source_id=record["raw_file_id"],
            location=record["name"],
            summary=summary,
            score=record.get("combined_score", 0.0),
            metadata=metadata,
        )


def _merge_table_results(
    keyword_results: list[dict[str, Any]],
    row_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge table metadata and row-hit legs by table-index id."""
    by_id: dict[str, dict[str, Any]] = {}
    for result in [*keyword_results, *row_results]:
        table_index_id = str(result["id"])
        current = by_id.get(table_index_id)
        if current is None or result.get("combined_score", 0.0) > current.get(
            "combined_score",
            0.0,
        ):
            by_id[table_index_id] = result
    return sorted(by_id.values(), key=lambda item: item.get("combined_score", 0.0), reverse=True)


def _row_match_score(plan_metadata: dict[str, Any], matched_rows: int) -> float:
    """Score direct row evidence ahead of schema-only table name matches."""
    score = 1.0 / 30.0
    if plan_metadata.get("identifiers"):
        score += 0.04
    if plan_metadata.get("predicates"):
        score += 0.02
    if plan_metadata.get("sort"):
        score += 0.01
    if matched_rows == 1:
        score += 0.01
    return score
