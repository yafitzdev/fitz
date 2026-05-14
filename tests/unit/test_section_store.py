# tests/unit/test_section_store.py
"""Tests for SectionStore — CRUD + search operations on section_index table."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore, _row_to_dict


@pytest.fixture
def mock_cm():
    cm = MagicMock()
    return cm


@pytest.fixture
def store(mock_cm):
    return SectionStore(mock_cm, "test_collection")


def _make_row(
    id_="sec1",
    raw_file_id="file1",
    title="Introduction",
    level=1,
    page_start=1,
    page_end=3,
    content="Section content here.",
    summary="A summary.",
    parent_section_id=None,
    position=0,
    metadata=None,
):
    """Create a tuple matching the section_index SELECT column order."""
    return (
        id_,
        raw_file_id,
        title,
        level,
        page_start,
        page_end,
        content,
        summary,
        parent_section_id,
        position,
        metadata or {},
    )


class TestRowToDict:
    def test_converts_tuple_to_dict(self):
        row = _make_row()
        result = _row_to_dict(row)
        assert result["id"] == "sec1"
        assert result["raw_file_id"] == "file1"
        assert result["title"] == "Introduction"
        assert result["level"] == 1
        assert result["page_start"] == 1
        assert result["page_end"] == 3
        assert result["content"] == "Section content here."
        assert result["summary"] == "A summary."
        assert result["parent_section_id"] is None
        assert result["position"] == 0
        assert result["metadata"] == {}

    def test_parses_json_string_metadata(self):
        row = _make_row(metadata='{"key": "value"}')
        result = _row_to_dict(row)
        assert result["metadata"] == {"key": "value"}

    def test_none_metadata_becomes_empty_dict(self):
        row = _make_row(metadata=None)
        result = _row_to_dict(row)
        assert result["metadata"] == {}


class TestUpsertBatch:
    def test_empty_batch_does_nothing(self, store, mock_cm):
        store.upsert_batch([])
        mock_cm.connection.assert_not_called()

    def test_upserts_sections(self, store, mock_cm):
        conn = MagicMock()
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        sections = [
            {
                "id": "sec1",
                "raw_file_id": "file1",
                "title": "Intro",
                "level": 1,
                "page_start": 1,
                "page_end": 2,
                "content": "Content here.",
                "summary": "A summary.",
                "parent_section_id": None,
                "position": 0,
                "metadata": {},
            }
        ]
        store.upsert_batch(sections)
        assert conn.execute.called
        assert conn.commit.called


class TestSearchBm25:
    def test_returns_results_with_bm25_score(self, store, mock_cm):
        # FTS5 bm25() returns negative numbers (lower=better); production code
        # flips the sign so downstream consumers treat higher as better.
        row = _make_row() + (-0.85,)
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [row]
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        results = store.search_bm25("introduction", limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Introduction"
        assert results[0]["bm25_score"] == 0.85


class TestGet:
    def test_returns_section_when_found(self, store, mock_cm):
        row = _make_row()
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = row
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = store.get("sec1")
        assert result is not None
        assert result["id"] == "sec1"

    def test_returns_none_when_not_found(self, store, mock_cm):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        result = store.get("nonexistent")
        assert result is None


class TestGetByFile:
    def test_returns_sections_ordered_by_position(self, store, mock_cm):
        rows = [
            _make_row(id_="sec1", position=0),
            _make_row(id_="sec2", position=1),
        ]
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = rows
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        results = store.get_by_file("file1")
        assert len(results) == 2
        assert results[0]["id"] == "sec1"
        assert results[1]["id"] == "sec2"


class TestDeleteByFile:
    def test_deletes_and_commits(self, store, mock_cm):
        conn = MagicMock()
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        store.delete_by_file("file1")
        assert conn.execute.called
        assert conn.commit.called
