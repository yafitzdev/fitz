# fitz_sage/tabular/store/__init__.py
"""Table storage backends."""

from __future__ import annotations

from fitz_sage.tabular.store.base import (
    StoredTable,
    TableStore,
    compress_csv,
    compute_hash,
    decompress_csv,
)
from fitz_sage.tabular.store.postgres import PostgresTableStore

__all__ = [
    "StoredTable",
    "TableStore",
    "PostgresTableStore",
    "compress_csv",
    "compute_hash",
    "decompress_csv",
    "get_table_store",
]


def get_table_store(collection: str) -> TableStore:
    """Return the PostgreSQL table store for a collection."""
    return PostgresTableStore(collection)
