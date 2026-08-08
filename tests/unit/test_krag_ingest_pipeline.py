# tests/unit/test_krag_ingest_pipeline.py
"""Tests for KRAG ingestion pipeline maintenance operations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline
from fitz_sage.engines.fitz_krag.ingestion.strategies.base import IngestResult


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


def test_code_without_finer_symbols_gets_file_module_address(tmp_path) -> None:
    """Top-level scripts remain filename-addressable instead of failing indexing."""
    source = tmp_path / "migration.js"
    source.write_text(
        "module.exports = {\n  up: query => query.run('ALTER TABLE jobs')\n};\n",
        encoding="utf-8",
    )
    strategy = MagicMock()
    strategy.extract.return_value = IngestResult()
    pipeline = KragIngestPipeline.__new__(KragIngestPipeline)
    pipeline._strategies = {"typescript": strategy}
    pipeline._raw_store = MagicMock()
    pipeline._symbol_store = MagicMock()
    pipeline._import_store = MagicMock()

    count = pipeline._parse_code_file("db/migration.js", source, "file-id")

    assert count == 1
    stored = pipeline._symbol_store.upsert_batch.call_args.args[0]
    assert stored[0]["name"] == "migration"
    assert stored[0]["qualified_name"] == "db.migration"
    assert stored[0]["kind"] == "module"
    assert stored[0]["start_line"] == 1
    assert stored[0]["end_line"] == 3


def test_table_parse_failure_preserves_the_concrete_cause(tmp_path) -> None:
    source = tmp_path / "broken.csv"
    source.write_text(",,\nTitle,,\n", encoding="utf-8")
    pipeline = KragIngestPipeline.__new__(KragIngestPipeline)

    with pytest.raises(
        ValueError,
        match=r"Could not parse table 'broken.csv': No headers",
    ) as caught:
        pipeline._parse_table_file("broken.csv", source, "file-id")
    assert str(tmp_path) not in str(caught.value)


def test_table_storage_failure_preserves_the_concrete_cause(tmp_path) -> None:
    source = tmp_path / "wide.csv"
    source.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    pipeline = KragIngestPipeline.__new__(KragIngestPipeline)
    pipeline._raw_store = MagicMock()
    pipeline._sqlite_table_store = MagicMock()
    pipeline._sqlite_table_store.store.side_effect = RuntimeError("too many columns")

    with pytest.raises(
        ValueError,
        match=r"Could not store table 'wide.csv': too many columns",
    ):
        pipeline._parse_table_file("wide.csv", source, "file-id")
