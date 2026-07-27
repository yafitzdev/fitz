# tests/unit/test_krag_table_search.py
"""Tests for TableSearchStrategy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.strategies.table_search import (
    TableSearchStrategy,
)
from fitz_sage.engines.fitz_krag.types import AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_strategy(
    keyword_results: list[dict] | None = None,
    *,
    sqlite_table_store: MagicMock | None = None,
) -> TableSearchStrategy:
    table_store = MagicMock(name="table_store")
    table_store.search_by_name.return_value = keyword_results or []
    table_store.get_by_table_id.return_value = None

    config = MagicMock(name="config")

    return TableSearchStrategy(table_store, config, sqlite_table_store)


def _make_table_record(
    record_id: str = "rec-001",
    table_id: str = "tbl_abc",
    name: str = "Sales Data",
    raw_file_id: str = "file1",
    columns: list[str] | None = None,
    row_count: int = 100,
    summary: str = "Sales records",
) -> dict:
    return {
        "id": record_id,
        "raw_file_id": raw_file_id,
        "table_id": table_id,
        "name": name,
        "columns": columns or ["product", "revenue"],
        "row_count": row_count,
        "summary": summary,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTableSearchStrategy:
    def test_retrieve_keyword_match(self):
        """Finds table by keyword search on column names."""
        record = _make_table_record(name="Revenue Report")
        strategy = _make_strategy(keyword_results=[record])

        addresses = strategy.retrieve("revenue", limit=5)

        assert len(addresses) == 1
        assert addresses[0].kind == AddressKind.TABLE
        assert addresses[0].location == "Revenue Report"

    def test_retrieve_returns_table_address(self):
        """Returns AddressKind.TABLE with correct metadata."""
        record = _make_table_record(
            record_id="rec-x",
            table_id="tbl_123",
            name="Test Table",
            columns=["a", "b", "c"],
            row_count=42,
        )
        strategy = _make_strategy(keyword_results=[record])

        addresses = strategy.retrieve("test", limit=5)

        assert len(addresses) == 1
        addr = addresses[0]
        assert addr.kind == AddressKind.TABLE
        assert addr.metadata["table_index_id"] == "rec-x"
        assert addr.metadata["table_id"] == "tbl_123"
        assert addr.metadata["name"] == "Test Table"
        assert addr.metadata["columns"] == ["a", "b", "c"]
        assert addr.metadata["row_count"] == 42
        assert addr.source_id == "file1"
        assert addr.score > 0

    def test_retrieve_empty(self):
        """Returns empty list when no tables match."""
        strategy = _make_strategy(keyword_results=[])

        addresses = strategy.retrieve("nonexistent", limit=5)

        assert addresses == []

    def test_retrieve_respects_limit(self):
        """Only returns up to limit addresses."""
        records = [
            _make_table_record(record_id=f"rec-{i}", table_id=f"tbl_{i}", name=f"Table {i}")
            for i in range(10)
        ]
        strategy = _make_strategy(keyword_results=records)

        addresses = strategy.retrieve("table", limit=3)

        assert len(addresses) == 3

    def test_retrieve_uses_bounded_row_lookup_for_table_obligation(self):
        """Row identifiers should surface the matching table, not only schema words."""
        record = _make_table_record(
            record_id="rec-vendors",
            table_id="tbl_vendors",
            name="Vendors",
            columns=["vendor_id", "vendor", "notice_days"],
            row_count=1,
        )
        sqlite_store = MagicMock(name="sqlite_table_store")
        sqlite_store.catalog.return_value = [{"table_id": "tbl_vendors"}]
        sqlite_store.scan_rows.return_value = (
            ["vendor_id", "vendor", "notice_days"],
            [["VEN-301", "MeridianAI", "75"]],
        )
        strategy = _make_strategy(keyword_results=[], sqlite_table_store=sqlite_store)
        strategy._table_store.get_by_table_id.return_value = record

        addresses = strategy.retrieve(
            "What notice days are recorded for vendor VEN-301?",
            limit=5,
            detection=SimpleNamespace(required_modalities=("table",)),
        )

        assert len(addresses) == 1
        assert addresses[0].location == "Vendors"
        assert addresses[0].metadata["row_match"]["matched_rows"] == 1
        assert addresses[0].metadata["rerank_text"] == (
            "Columns: vendor_id | vendor | notice_days\nRow: VEN-301 | MeridianAI | 75"
        )

    def test_retrieve_searches_row_values_without_table_classification(self):
        """Ordinary lookups must find table rows even when query profiling misses the modality."""
        record = _make_table_record(
            record_id="rec-rollouts",
            table_id="tbl_rollouts",
            name="Rollout Matrix",
            columns=["feature", "region", "release", "status"],
            row_count=2,
        )
        sqlite_store = MagicMock(name="sqlite_table_store")
        sqlite_store.search_rows_bm25.return_value = [
            {
                "table_id": "tbl_rollouts",
                "rank": 1,
                "bm25_score": 3.2,
                "matched_rows": 1,
                "row_numbers": [1],
                "row_texts": ["token_rotation eu 2026.05 enabled"],
            }
        ]
        strategy = _make_strategy(keyword_results=[], sqlite_table_store=sqlite_store)
        strategy._table_store.get_by_table_id.return_value = record

        addresses = strategy.retrieve(
            "Which release enabled token rotation in the EU region?",
            limit=5,
            detection=SimpleNamespace(required_modalities=("symbol", "section")),
        )

        assert [address.location for address in addresses] == ["Rollout Matrix"]
        assert addresses[0].metadata["row_search"]["matched_rows"] == 1
        assert addresses[0].metadata["rerank_text"] == (
            "Columns: feature | region | release | status\nRow: token_rotation eu 2026.05 enabled"
        )
        sqlite_store.scan_rows.assert_not_called()

    def test_retrieve_uses_full_table_exact_identifier_lookup(self):
        """An exact ID beyond the bounded scan should still surface its table."""
        record = _make_table_record(
            record_id="rec-assets",
            table_id="tbl_assets",
            name="Assets",
            columns=["asset_id", "owner"],
            row_count=900,
        )
        sqlite_store = MagicMock(name="sqlite_table_store")
        sqlite_store.catalog.return_value = [{"table_id": "tbl_assets"}]
        sqlite_store.find_rows_by_identifiers.return_value = (
            ["asset_id", "owner"],
            [["AX-156", "Platform"]],
        )
        strategy = _make_strategy(keyword_results=[], sqlite_table_store=sqlite_store)
        strategy._table_store.get_by_table_id.return_value = record

        addresses = strategy.retrieve("Who owns AX-156?", limit=5)

        assert len(addresses) == 1
        assert addresses[0].metadata["row_match"]["exact_identifier_lookup"] is True
        sqlite_store.scan_rows.assert_not_called()

    def test_retrieve_scans_rows_for_explicit_structured_record_shape(self):
        """Record-property questions warrant a bounded row scan without an exact ID."""
        record = _make_table_record(
            record_id="rec-edges",
            table_id="tbl_edges",
            name="Edge Records",
            columns=[
                "edge_id",
                "station",
                "status",
                "latency_ms",
                "retries",
                "owner",
                "release",
            ],
            row_count=2,
        )
        sqlite_store = MagicMock(name="sqlite_table_store")
        sqlite_store.search_rows_bm25.return_value = []
        sqlite_store.catalog.return_value = [{"table_id": "tbl_edges"}]
        sqlite_store.scan_rows.return_value = (
            record["columns"],
            [
                ["EDGE-106", "delta", "fail", "390", "31", "Rhea", "REL-2026.04"],
                ["EDGE-107", "delta", "pass", "210", "4", "Ivo", "REL-2026.04"],
            ],
        )
        strategy = _make_strategy(keyword_results=[], sqlite_table_store=sqlite_store)
        strategy._table_store.get_by_table_id.return_value = record

        addresses = strategy.retrieve(
            "Who owns the failed delta edge record?",
            limit=5,
        )

        assert [address.location for address in addresses] == ["Edge Records"]
        assert addresses[0].metadata["row_match"]["matched_rows"] >= 1
        sqlite_store.scan_rows.assert_called_once_with("tbl_edges", limit=500)

    def test_prose_record_reference_does_not_trigger_row_scan(self):
        """The noun 'record' alone is not sufficient structured-data intent."""
        sqlite_store = MagicMock(name="sqlite_table_store")
        sqlite_store.search_rows_bm25.return_value = []
        strategy = _make_strategy(keyword_results=[], sqlite_table_store=sqlite_store)

        strategy.retrieve(
            "What does the records retention policy require?",
            limit=5,
        )

        sqlite_store.catalog.assert_not_called()
        sqlite_store.scan_rows.assert_not_called()
