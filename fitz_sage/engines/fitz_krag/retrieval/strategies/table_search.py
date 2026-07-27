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
import re
from typing import TYPE_CHECKING, Any

from fitz_sage.core.identifiers import exact_identifiers
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
_MAX_RERANK_ROWS = 5
_ROW_SCAN_LIMIT = 500
_ROW_REFERENCE_PATTERN = re.compile(r"\b(?:entry|entries|record|records|row|rows)\b", re.I)
_ROW_ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:assigned|count|date|earliest|failed|failure|highest|latest|lowest|"
    r"owner|owned|owns|passed|release|responsible|state|status|total|value|version)\b",
    re.I,
)


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
        indexed_row_results = self._indexed_row_results(query, limit=fetch_limit)
        row_results = self._row_results(query, limit=fetch_limit, profile=detection)

        scored = _merge_table_results(
            rrf_score(keyword_results),
            indexed_row_results,
            row_results,
        )[:limit]

        return [self._to_address(r) for r in scored]

    def _indexed_row_results(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return table records surfaced by BM25 over concrete row values."""
        search_rows = getattr(self._sqlite_table_store, "search_rows_bm25", None)
        get_by_table_id = getattr(self._table_store, "get_by_table_id", None)
        if not callable(search_rows) or not callable(get_by_table_id):
            return []
        hits = search_rows(query, limit=limit)
        if not isinstance(hits, list):
            return []

        results: list[dict[str, Any]] = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            record = get_by_table_id(str(hit.get("table_id") or ""))
            if not record:
                continue
            metadata = dict(record.get("metadata") or {})
            metadata["row_search"] = dict(hit)
            results.append(
                {
                    **record,
                    "metadata": metadata,
                    "combined_score": _indexed_row_score(hit),
                }
            )
        return results

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
        find_rows_by_identifiers = getattr(
            self._sqlite_table_store,
            "find_rows_by_identifiers",
            None,
        )
        get_by_table_id = getattr(self._table_store, "get_by_table_id", None)
        if not callable(catalog) or not callable(get_by_table_id):
            return []

        identifiers = tuple(exact_identifiers(query))
        matches: list[dict[str, Any]] = []
        for table in catalog():
            table_id = str(table.get("table_id") or "")
            if not table_id:
                continue
            row_data: tuple[list[str], list[list[Any]]] | None = None
            exact_identifier_lookup = False
            if identifiers and callable(find_rows_by_identifiers):
                candidate = find_rows_by_identifiers(
                    table_id,
                    identifiers,
                    limit=max(limit, len(identifiers)),
                )
                if _is_row_data(candidate):
                    row_data = candidate
                    exact_identifier_lookup = True
            if row_data is None and callable(scan_rows):
                candidate = scan_rows(table_id, limit=_ROW_SCAN_LIMIT)
                if _is_row_data(candidate):
                    row_data = candidate
            if row_data is None:
                continue
            columns, rows = row_data
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
                "exact_identifier_lookup": exact_identifier_lookup,
                "matched_rows": len(selected_rows),
                "plan": plan.metadata,
                "rerank_rows": selected_rows[:_MAX_RERANK_ROWS],
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
        if exact_identifiers(query):
            return True
        return bool(_ROW_REFERENCE_PATTERN.search(query) and _ROW_ATTRIBUTE_PATTERN.search(query))

    def _to_address(self, record: dict[str, Any]) -> Address:
        """Convert a table store row to an Address."""
        columns = list(record["columns"])
        summary = record.get("summary") or (
            f"Table {record['name']} columns: {', '.join(columns)}. Rows: {record['row_count']}."
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
        row_context = _table_row_context(metadata, columns)
        if row_context:
            existing_context = metadata.get("rerank_text")
            metadata["rerank_text"] = _join_distinct_text(existing_context, row_context)
        return Address(
            kind=AddressKind.TABLE,
            source_id=record["raw_file_id"],
            location=record["name"],
            summary=summary,
            score=record.get("combined_score", 0.0),
            metadata=metadata,
        )


def _merge_table_results(
    *result_sets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge table metadata and row-hit legs by table-index id."""
    by_id: dict[str, dict[str, Any]] = {}
    for result_set in result_sets:
        for result in result_set:
            table_index_id = str(result["id"])
            current = by_id.get(table_index_id)
            if current is None:
                by_id[table_index_id] = result
                continue
            if result.get("combined_score", 0.0) > current.get("combined_score", 0.0):
                winner, other = result, current
            else:
                winner, other = current, result
            merged = dict(winner)
            metadata = dict(other.get("metadata") or {})
            metadata.update(dict(winner.get("metadata") or {}))
            merged["metadata"] = metadata
            by_id[table_index_id] = merged
    return sorted(by_id.values(), key=lambda item: item.get("combined_score", 0.0), reverse=True)


def _table_row_context(metadata: dict[str, Any], columns: list[str]) -> str:
    """Return a bounded, source-faithful preview of rows found during retrieval."""
    row_search = metadata.get("row_search")
    if isinstance(row_search, dict):
        row_texts = row_search.get("row_texts")
        if isinstance(row_texts, list):
            values = [str(value).strip() for value in row_texts if str(value).strip()]
            if values:
                return "\n".join(
                    [f"Columns: {' | '.join(columns)}", *(f"Row: {value}" for value in values)]
                )

    row_match = metadata.get("row_match")
    if not isinstance(row_match, dict):
        return ""
    rows = row_match.get("rerank_rows")
    if not isinstance(rows, list):
        return ""
    rendered_rows = [
        " | ".join(str(value) for value in row) for row in rows if isinstance(row, list) and row
    ]
    if not rendered_rows:
        return ""
    return "\n".join(
        [f"Columns: {' | '.join(columns)}", *(f"Row: {value}" for value in rendered_rows)]
    )


def _join_distinct_text(existing: Any, additional: str) -> str:
    """Append retrieval context without duplicating an existing identical value."""
    if not isinstance(existing, str) or not existing.strip():
        return additional
    if additional in existing:
        return existing
    return f"{existing.strip()}\n{additional}"


def _indexed_row_score(hit: dict[str, Any]) -> float:
    """Score row-index matches ahead of metadata-only matches."""
    try:
        rank = max(1, int(hit.get("rank", 1)))
    except (TypeError, ValueError):
        rank = 1
    return (1.0 / 30.0) + (1.0 / (60 + rank))


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


def _is_row_data(value: Any) -> bool:
    """Return whether a store result has the expected columns-and-rows shape."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], list)
        and isinstance(value[1], list)
    )
