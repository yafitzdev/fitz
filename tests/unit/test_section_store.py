# tests/unit/test_section_store.py
"""Tests for SectionStore — CRUD + search operations on section_index table."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.ingestion.section_store import (
    CORPUS_SUMMARY_SCHEMA_VERSION,
    SectionStore,
    _row_to_dict,
)


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
    entities=None,
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
        entities or [],
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
        assert result["entities"] == []
        assert result["metadata"] == {}

    def test_parses_json_entity_column(self):
        row = _make_row(
            entities='[{"name": "Acme", "type": "org"}]',
        )
        result = _row_to_dict(row)
        assert result["entities"] == [{"name": "Acme", "type": "org"}]

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
        rank_cursor = MagicMock()
        rank_cursor.fetchall.return_value = [(17, -0.85)]
        section_cursor = MagicMock()
        section_cursor.fetchall.return_value = [(17, *_make_row())]
        conn = MagicMock()
        conn.execute.side_effect = [rank_cursor, section_cursor]
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        results = store.search_bm25("introduction", limit=10)
        assert len(results) == 1
        assert results[0]["title"] == "Introduction"
        assert results[0]["bm25_score"] == 0.85

    def test_preserves_fts_rank_order_across_batch_materialization(self, store, mock_cm):
        rank_cursor = MagicMock()
        rank_cursor.fetchall.return_value = [(22, -0.9), (11, -0.8)]
        section_cursor = MagicMock()
        section_cursor.fetchall.return_value = [
            (11, *_make_row(id_="second")),
            (22, *_make_row(id_="first")),
        ]
        conn = MagicMock()
        conn.execute.side_effect = [rank_cursor, section_cursor]
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        results = store.search_bm25("introduction", limit=10)

        assert [result["id"] for result in results] == ["first", "second"]


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


class TestGetCorpusSummaries:
    def test_filters_stale_corpus_summary_schema(self, tmp_path: Path):
        from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
        from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
        from fitz_sage.storage.config import StorageConfig
        from fitz_sage.storage.sqlite import SqliteConnectionManager

        cm = SqliteConnectionManager(StorageConfig(storage_path=tmp_path))
        collection = "test_collection"
        ensure_schema(cm, collection)
        raw_store = RawFileStore(cm, collection)
        store = SectionStore(cm, collection)
        raw_store.upsert("old-file", "__corpus_summary__", "old", "old-hash", ".md", 3)
        raw_store.upsert("new-file", "__corpus_summary__", "new", "new-hash", ".md", 3)
        store.upsert_batch(
            [
                {
                    "id": "old-summary",
                    "raw_file_id": "old-file",
                    "title": "Corpus Overview",
                    "level": 0,
                    "content": "old cybersecurity summary",
                    "summary": "old cybersecurity summary",
                    "position": 0,
                    "metadata": {"is_corpus_summary": True},
                },
                {
                    "id": "new-summary",
                    "raw_file_id": "new-file",
                    "title": "Corpus Overview",
                    "level": 0,
                    "content": "fresh corpus summary",
                    "summary": "fresh corpus summary",
                    "position": 1,
                    "metadata": {
                        "is_corpus_summary": True,
                        "corpus_summary_schema": CORPUS_SUMMARY_SCHEMA_VERSION,
                    },
                },
            ]
        )

        results = store.get_corpus_summaries()

        assert [r["id"] for r in results] == ["new-summary"]
        assert results[0]["content"] == "fresh corpus summary"

    def test_bm25_excludes_synthetic_corpus_summaries(self, tmp_path: Path):
        from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
        from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
        from fitz_sage.storage.config import StorageConfig
        from fitz_sage.storage.sqlite import SqliteConnectionManager

        cm = SqliteConnectionManager(StorageConfig(storage_path=tmp_path))
        collection = "test_collection"
        ensure_schema(cm, collection)
        raw_store = RawFileStore(cm, collection)
        store = SectionStore(cm, collection)
        raw_store.upsert("corpus-file", "__corpus_summary__", "corpus", "corpus-hash", ".md", 6)
        raw_store.upsert("doc-file", "docs/real.md", "corpus", "doc-hash", ".md", 6)
        store.upsert_batch(
            [
                {
                    "id": "corpus-summary",
                    "raw_file_id": "corpus-file",
                    "title": "Corpus Overview",
                    "level": 0,
                    "content": "corpus overview text",
                    "summary": "corpus overview text",
                    "position": 0,
                    "metadata": {
                        "is_corpus_summary": True,
                        "corpus_summary_schema": CORPUS_SUMMARY_SCHEMA_VERSION,
                    },
                },
                {
                    "id": "real-section",
                    "raw_file_id": "doc-file",
                    "title": "Corpus Notes",
                    "level": 1,
                    "content": "corpus notes text",
                    "summary": "corpus notes text",
                    "position": 1,
                    "metadata": {},
                },
            ]
        )

        results = store.search_bm25("corpus", limit=10)

        assert [r["id"] for r in results] == ["real-section"]


class TestDeleteByFile:
    def test_deletes_and_commits(self, store, mock_cm):
        conn = MagicMock()
        mock_cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
        mock_cm.connection.return_value.__exit__ = MagicMock(return_value=False)

        store.delete_by_file("file1")
        assert conn.execute.called
        assert conn.commit.called
