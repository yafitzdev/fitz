# tests/unit/tabular/test_store.py
"""Tests for table storage backends."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator

from fitz_sage.tabular.store.base import compress_csv, compute_hash, decompress_csv
from fitz_sage.tabular.store.sqlite import SqliteTableStore


class _InMemoryConnectionManager:
    def __init__(self) -> None:
        self.connection_object = sqlite3.connect(":memory:")

    def start(self) -> None:
        pass

    @contextmanager
    def connection(self, collection: str) -> Generator[sqlite3.Connection, None, None]:
        del collection
        yield self.connection_object


def _in_memory_table_store() -> SqliteTableStore:
    store = SqliteTableStore.__new__(SqliteTableStore)
    store.collection = "test"
    store._manager = _InMemoryConnectionManager()
    store._schema_initialized = False
    return store


class TestCompressionUtils:
    """Tests for compression utilities."""

    def test_compute_hash_deterministic(self):
        """Test that hash is deterministic."""
        columns = ["a", "b", "c"]
        rows = [["1", "2", "3"], ["4", "5", "6"]]

        hash1 = compute_hash(columns, rows)
        hash2 = compute_hash(columns, rows)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_hash_different_content(self):
        """Test that different content produces different hash."""
        columns = ["a", "b"]
        rows1 = [["1", "2"]]
        rows2 = [["3", "4"]]

        hash1 = compute_hash(columns, rows1)
        hash2 = compute_hash(columns, rows2)

        assert hash1 != hash2

    def test_compress_decompress_roundtrip(self):
        """Test compression and decompression roundtrip."""
        columns = ["Name", "Age", "City"]
        rows = [
            ["Alice", "30", "New York"],
            ["Bob", "25", "Los Angeles"],
            ["Carol", "35", "Chicago"],
        ]

        compressed = compress_csv(columns, rows)
        decompressed_cols, decompressed_rows = decompress_csv(compressed)

        assert decompressed_cols == columns
        assert decompressed_rows == rows

    def test_compression_reduces_size(self):
        """Test that compression actually reduces size."""
        columns = ["A", "B", "C", "D", "E"]
        rows = [[f"value{i}{j}" for j in range(5)] for i in range(100)]

        # Calculate uncompressed size
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        writer.writerows(rows)
        uncompressed_size = len(buffer.getvalue().encode())

        compressed = compress_csv(columns, rows)

        # Compressed should be significantly smaller
        assert len(compressed) < uncompressed_size / 2

    def test_decompress_empty_table(self):
        """Test decompressing empty table."""
        columns = ["A", "B"]
        rows = []

        compressed = compress_csv(columns, rows)
        decompressed_cols, decompressed_rows = decompress_csv(compressed)

        assert decompressed_cols == columns
        assert decompressed_rows == []


def test_exact_identifier_lookup_searches_beyond_scan_limit() -> None:
    store = _in_memory_table_store()
    rows = [[f"ROW-{index}", "ordinary"] for index in range(700)]
    rows[610] = ["AX-156B", "neighbor"]
    rows[611] = ["AX_156", "separator variant"]
    rows[612] = ["AX 156", "spaced variant"]
    rows[650] = ["AX-156", "exact"]
    store.store("assets", ["asset_id", "owner"], rows, "assets.csv")

    result = store.find_rows_by_identifiers("assets", ["AX-156"])

    assert result == (["asset_id", "owner"], [["AX-156", "exact"]])
