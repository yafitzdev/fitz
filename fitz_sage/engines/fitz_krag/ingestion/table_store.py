# fitz_sage/engines/fitz_krag/ingestion/table_store.py
"""CRUD operations for krag_table_index table."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}table_index"


class TableStore:
    """CRUD for the table metadata index."""

    def __init__(self, connection_manager: "PostgresConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, tables: list[dict[str, Any]]) -> None:
        """Insert or update a batch of table metadata records."""
        if not tables:
            return

        sql = f"""
            INSERT INTO {TABLE}
                ("id", "raw_file_id", "table_id", "name", "columns", "row_count",
                 "summary", "metadata")
            VALUES
                (%s, %s, %s, %s, %s, %s,
                 %s, %s::jsonb)
            ON CONFLICT ("id") DO UPDATE SET
                "raw_file_id" = EXCLUDED."raw_file_id",
                "table_id" = EXCLUDED."table_id",
                "name" = EXCLUDED."name",
                "columns" = EXCLUDED."columns",
                "row_count" = EXCLUDED."row_count",
                "summary" = EXCLUDED."summary",
                "metadata" = EXCLUDED."metadata"
        """
        with self._cm.connection(self._collection) as conn:
            for tbl in tables:
                conn.execute(
                    sql,
                    (
                        tbl["id"],
                        tbl["raw_file_id"],
                        tbl["table_id"],
                        tbl["name"],
                        tbl["columns"],
                        tbl["row_count"],
                        tbl.get("summary"),
                        json.dumps(tbl.get("metadata", {})),
                    ),
                )
            conn.commit()

    def search_by_name(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """ILIKE search on name and column names using individual query words.

        Extracts significant words from the query (length >= 3) and matches
        each against table names and column names.  Basic plural stemming
        (strip trailing 's') ensures 'managers' matches 'manager_id'.
        """
        keywords = set()
        for word in query.lower().split():
            word = word.strip("?.,!;:()")
            if len(word) < 3:
                continue
            keywords.add(word)
            if word.endswith("s") and len(word) > 4:
                keywords.add(word[:-1])

        if not keywords:
            return []

        conditions = []
        params: list[object] = []
        for kw in keywords:
            pattern = f"%{kw}%"
            conditions.append(
                '"name" ILIKE %s OR EXISTS '
                '(SELECT 1 FROM unnest("columns") AS col WHERE col ILIKE %s)'
            )
            params.extend([pattern, pattern])

        where_clause = " OR ".join(f"({c})" for c in conditions)
        sql = f"""
            SELECT "id", "raw_file_id", "table_id", "name", "columns", "row_count",
                   "summary", "metadata"
            FROM {TABLE}
            WHERE {where_clause}
            LIMIT %s
        """
        params.append(limit)
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get(self, table_index_id: str) -> dict[str, Any] | None:
        """Get a single table record by ID."""
        sql = f"""
            SELECT "id", "raw_file_id", "table_id", "name", "columns", "row_count",
                   "summary", "metadata"
            FROM {TABLE} WHERE "id" = %s
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (table_index_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        """Get all table records for a raw file."""
        sql = f"""
            SELECT "id", "raw_file_id", "table_id", "name", "columns", "row_count",
                   "summary", "metadata"
            FROM {TABLE}
            WHERE "raw_file_id" = %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_summary(self, table_index_id: str, summary: str) -> None:
        """Update summary for a single table record."""
        sql = f'UPDATE {TABLE} SET "summary" = %s WHERE "id" = %s'
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (summary, table_index_id))
            conn.commit()

    def delete_by_file(self, raw_file_id: str) -> None:
        """Delete table metadata when source file is removed."""
        sql = f'DELETE FROM {TABLE} WHERE "raw_file_id" = %s'
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (raw_file_id,))
            conn.commit()


def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a query row to dict."""
    meta = row[7]
    if isinstance(meta, str):
        meta = json.loads(meta)
    elif meta is None:
        meta = {}
    columns = list(row[4]) if row[4] else []
    return {
        "id": row[0],
        "raw_file_id": row[1],
        "table_id": row[2],
        "name": row[3],
        "columns": columns,
        "row_count": row[5],
        "summary": row[6],
        "metadata": meta,
    }
