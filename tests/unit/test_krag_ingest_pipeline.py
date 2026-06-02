# tests/unit/test_krag_ingest_pipeline.py
"""Tests for KRAG ingestion pipeline maintenance operations."""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline


def test_delete_files_not_in_paths_removes_stale_rows_only() -> None:
    """Re-pointing a collection deletes stale files while preserving corpus metadata."""
    pipeline = KragIngestPipeline.__new__(KragIngestPipeline)
    pipeline._raw_store = MagicMock()
    pipeline._raw_store.list_ids_by_path.return_value = {
        "README.md": "keep-id",
        ".fitz/collections/rag/source_dir.txt": "internal-id",
        "__corpus_summary__": "corpus-id",
    }
    pipeline._table_store = MagicMock()
    pipeline._table_store.get_by_file.return_value = []
    pipeline._sqlite_table_store = None

    deleted = pipeline.delete_files_not_in_paths({"README.md"})

    assert deleted == 1
    pipeline._raw_store.delete.assert_called_once_with("internal-id")
    pipeline._table_store.delete_by_file.assert_called_once_with("internal-id")
