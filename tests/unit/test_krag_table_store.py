# tests/unit/test_krag_table_store.py
"""Tests for KRAG TableStore (table metadata index)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.ingestion.table_store import TABLE, TableStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> tuple[TableStore, MagicMock]:
    """Create a TableStore with a mocked connection manager."""
    cm = MagicMock(name="connection_manager")
    store = TableStore(cm, "test_collection")
    return store, cm


def _make_table_record(
    table_id: str = "tbl_abc123",
    name: str = "Sales Data",
    columns: list[str] | None = None,
    row_count: int = 100,
    summary: str = "Sales records with revenue data",
    raw_file_id: str = "file1",
    record_id: str = "rec-001",
) -> dict:
    return {
        "id": record_id,
        "raw_file_id": raw_file_id,
        "table_id": table_id,
        "name": name,
        "columns": columns or ["product", "revenue", "date"],
        "row_count": row_count,
        "summary": summary,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# TestUpsertBatch
# ---------------------------------------------------------------------------


class TestUpsertBatch:
    def test_upsert_batch_stores_records(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        records = [_make_table_record()]

        store.upsert_batch(records)

        assert conn.execute.call_count == 1
        conn.commit.assert_called_once()
        args = conn.execute.call_args[0]
        assert TABLE in args[0]
        assert args[1][0] == "rec-001"  # id
        assert args[1][2] == "tbl_abc123"  # table_id
        assert args[1][3] == "Sales Data"  # name

    def test_upsert_batch_empty(self):
        store, cm = _make_store()
        store.upsert_batch([])
        cm.connection.assert_not_called()

    def test_upsert_batch_multiple(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        records = [
            _make_table_record(record_id="r1", table_id="t1"),
            _make_table_record(record_id="r2", table_id="t2"),
        ]

        store.upsert_batch(records)

        assert conn.execute.call_count == 2
        conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# TestSearchByName
# ---------------------------------------------------------------------------


class TestSearchByName:
    def test_search_by_name(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = [
            (
                "rec-001",
                "file1",
                "tbl_abc",
                "Sales Data",
                ["product", "revenue"],
                100,
                "Summary",
                "{}",
            ),
        ]

        results = store.search_by_name("revenue", limit=5)

        assert len(results) == 1
        assert results[0]["name"] == "Sales Data"
        assert results[0]["columns"] == ["product", "revenue"]
        assert results[0]["table_id"] == "tbl_abc"

    def test_search_by_name_empty(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = []

        results = store.search_by_name("nonexistent")

        assert results == []


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = (
            "rec-001",
            "file1",
            "tbl_abc",
            "Sales",
            ["col1", "col2"],
            100,
            "Summary",
            "{}",
        )

        result = store.get("rec-001")

        assert result is not None
        assert result["id"] == "rec-001"
        assert result["columns"] == ["col1", "col2"]

    def test_get_missing(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = None

        result = store.get("nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# TestGetByFile
# ---------------------------------------------------------------------------


class TestGetByFile:
    def test_get_by_file(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value
        conn.execute.return_value.fetchall.return_value = [
            ("rec-001", "file1", "tbl_abc", "Sales", ["col1"], 100, "Summary", "{}"),
            ("rec-002", "file1", "tbl_def", "Revenue", ["col2"], 50, "Other", "{}"),
        ]

        results = store.get_by_file("file1")

        assert len(results) == 2
        assert results[0]["table_id"] == "tbl_abc"
        assert results[1]["table_id"] == "tbl_def"


# ---------------------------------------------------------------------------
# TestDeleteByFile
# ---------------------------------------------------------------------------


class TestDeleteByFile:
    def test_delete_by_file(self):
        store, cm = _make_store()
        conn = cm.connection.return_value.__enter__.return_value

        store.delete_by_file("file1")

        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        assert "DELETE" in args[0]
        assert args[1] == ("file1",)
        conn.commit.assert_called_once()


