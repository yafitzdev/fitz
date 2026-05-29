# fitz_sage/tabular/store/base.py
"""TableStore protocol and utility functions."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class StoredTable:
    """Table data retrieved from store."""

    table_id: str
    hash: str
    columns: list[str]
    rows: list[list[str]]
    row_count: int
    source_file: str = ""


@runtime_checkable
class TableStore(Protocol):
    """Protocol for table storage backends."""

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
    ) -> str:
        """
        Store table, return content hash.

        Args:
            table_id: Unique identifier for the table
            columns: Column headers
            rows: Data rows (list of lists)
            source_file: Original source file path

        Returns:
            Content hash for cache invalidation
        """
        ...

    def retrieve(self, table_id: str) -> StoredTable | None:
        """
        Retrieve table by ID.

        Args:
            table_id: Table identifier

        Returns:
            StoredTable if found, None otherwise
        """
        ...

    def delete(self, table_id: str) -> None:
        """Delete a table."""
        ...


def compute_hash(columns: list[str], rows: list[list[str]]) -> str:
    """
    Compute deterministic hash of table content.

    Uses SHA-256, truncated to 16 chars for reasonable uniqueness
    while keeping IDs manageable.
    """
    content = ",".join(columns) + "\n"
    for row in rows:
        content += ",".join(str(cell) for cell in row) + "\n"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compress_csv(columns: list[str], rows: list[list[str]]) -> bytes:
    """
    Compress table as gzipped CSV.

    CSV format is ~40% smaller than JSON for tabular data.
    Gzip adds another ~5x reduction.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    return gzip.compress(buffer.getvalue().encode())


def decompress_csv(data: bytes) -> tuple[list[str], list[list[str]]]:
    """
    Decompress gzipped CSV to columns and rows.

    Returns:
        Tuple of (columns, rows)
    """
    text = gzip.decompress(data).decode()
    reader = csv.reader(io.StringIO(text))
    all_rows = list(reader)
    if not all_rows:
        return [], []
    return all_rows[0], all_rows[1:]  # headers, data rows
