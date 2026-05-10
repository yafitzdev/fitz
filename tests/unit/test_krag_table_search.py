# tests/unit/test_krag_table_search.py
"""Tests for TableSearchStrategy."""

from __future__ import annotations

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
    semantic_results: list[dict] | None = None,
    embed_vector: list[float] | None = None,
    table_keyword_weight: float = 0.4,
    table_semantic_weight: float = 0.6,
) -> TableSearchStrategy:
    table_store = MagicMock(name="table_store")
    table_store.search_by_name.return_value = keyword_results or []
    table_store.search_by_vector.return_value = semantic_results or []

    embedder = MagicMock(name="embedder")
    embedder.embed.return_value = embed_vector or [0.1, 0.2, 0.3]

    config = MagicMock(name="config")
    config.table_keyword_weight = table_keyword_weight
    config.table_semantic_weight = table_semantic_weight

    return TableSearchStrategy(table_store, config)


def _make_table_record(
    record_id: str = "rec-001",
    table_id: str = "tbl_abc",
    name: str = "Sales Data",
    raw_file_id: str = "file1",
    columns: list[str] | None = None,
    row_count: int = 100,
    summary: str = "Sales records",
    score: float | None = None,
) -> dict:
    d = {
        "id": record_id,
        "raw_file_id": raw_file_id,
        "table_id": table_id,
        "name": name,
        "columns": columns or ["product", "revenue"],
        "row_count": row_count,
        "summary": summary,
        "metadata": {},
    }
    if score is not None:
        d["score"] = score
    return d


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
        strategy = _make_strategy(keyword_results=[], semantic_results=[])

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

    def test_retrieve_semantic_failure_falls_back(self):
        """When semantic search fails, keyword results still returned."""
        record = _make_table_record(name="Data")
        table_store = MagicMock()
        table_store.search_by_name.return_value = [record]
        table_store.search_by_vector.return_value = []
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("embedding failed")
        config = MagicMock()
        config.table_keyword_weight = 0.4
        config.table_semantic_weight = 0.6

        strategy = TableSearchStrategy(table_store, config)
        addresses = strategy.retrieve("data", limit=5)

        assert len(addresses) == 1
        assert addresses[0].location == "Data"
