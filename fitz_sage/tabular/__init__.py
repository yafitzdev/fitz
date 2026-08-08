"""Native table parsing and SQLite row storage used by KRAG."""

from fitz_sage.tabular.parser import (
    SUPPORTED_EXTENSIONS,
    ParsedTableFile,
    can_parse,
    get_sample_rows,
    parse_csv,
)
from fitz_sage.tabular.store import SqliteTableStore, StoredTable, TableStore, get_table_store

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ParsedTableFile",
    "SqliteTableStore",
    "StoredTable",
    "TableStore",
    "can_parse",
    "get_sample_rows",
    "get_table_store",
    "parse_csv",
]
