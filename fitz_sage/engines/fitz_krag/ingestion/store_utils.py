# fitz_sage/engines/fitz_krag/ingestion/store_utils.py
"""Shared SQLite helpers for the KRAG index stores (symbol / section / table).

These stores share FTS-query building, JSON-column decoding, per-file deletion,
and enrichment writeback. The ``table`` argument is always a trusted module
constant (never user input), so f-string interpolation into SQL is safe.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager


def build_fts_query(query: str) -> str | None:
    """OR-join the alphanumeric words of a query into FTS5 MATCH syntax.

    FTS5 MATCH has its own syntax, so free-form user text is sanitized to
    alphanumeric tokens. Returns None for empty input so callers short-circuit.
    """
    words = [w for w in re.findall(r"\w+", query) if w]
    if not words:
        return None
    return " OR ".join(f'"{word}"' for word in words)


def decode_json(value: Any, default: Any) -> Any:
    """Decode a possibly-JSON-encoded DB column, falling back to ``default``.

    Already-parsed list/dict values pass through; None and parse errors yield
    the default.
    """
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def delete_by_file(
    cm: "SqliteConnectionManager", collection: str, table: str, raw_file_id: str
) -> None:
    """Delete all rows for a raw file from an index table."""
    with cm.connection(collection) as conn:
        conn.execute(f"DELETE FROM {table} WHERE raw_file_id = ?", (raw_file_id,))
        conn.commit()


def has_rows(cm: SqliteConnectionManager, collection: str, table: str) -> bool:
    """Return whether an index table contains at least one row."""
    with cm.connection(collection) as conn:
        return conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None


def update_entities_by_file(
    cm: "SqliteConnectionManager",
    collection: str,
    table: str,
    enriched_dicts: list[dict[str, Any]],
) -> None:
    """Write entity and hierarchy metadata back onto index rows by id."""
    sql = f"UPDATE {table} SET entities = ?, metadata = ? WHERE id = ?"
    with cm.connection(collection) as conn:
        for item in enriched_dicts:
            conn.execute(
                sql,
                (
                    json.dumps(item.get("entities", [])),
                    json.dumps(item.get("metadata", {})),
                    item["id"],
                ),
            )
        conn.commit()
