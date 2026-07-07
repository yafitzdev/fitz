# fitz_sage/engines/fitz_krag/ingestion/table_store.py
"""CRUD operations for krag_table_index table."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion import store_utils
from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}table_index"


class TableStore:
    """CRUD for the table metadata index."""

    def __init__(self, connection_manager: "SqliteConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, tables: list[dict[str, Any]]) -> None:
        if not tables:
            return

        sql = f"""
            INSERT INTO {TABLE}
                (id, raw_file_id, table_id, name, columns, row_count,
                 summary, metadata)
            VALUES
                (?, ?, ?, ?, ?, ?,
                 ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                raw_file_id = excluded.raw_file_id,
                table_id = excluded.table_id,
                name = excluded.name,
                columns = excluded.columns,
                row_count = excluded.row_count,
                summary = excluded.summary,
                metadata = excluded.metadata
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
                        json.dumps(list(tbl["columns"])),
                        tbl["row_count"],
                        tbl.get("summary"),
                        json.dumps(tbl.get("metadata", {})),
                    ),
                )
            conn.commit()

    def search_by_name(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Substring match on name and column names using significant query words."""
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
                "name LIKE ? COLLATE NOCASE "
                "OR EXISTS (SELECT 1 FROM json_each(columns) c "
                "WHERE c.value LIKE ? COLLATE NOCASE)"
            )
            params.extend([pattern, pattern])

        where_clause = " OR ".join(f"({c})" for c in conditions)
        sql = f"""
            SELECT id, raw_file_id, table_id, name, columns, row_count,
                   summary, metadata
            FROM {TABLE}
            WHERE {where_clause}
            LIMIT ?
        """
        params.append(limit)
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get(self, table_index_id: str) -> dict[str, Any] | None:
        sql = f"""
            SELECT id, raw_file_id, table_id, name, columns, row_count,
                   summary, metadata
            FROM {TABLE} WHERE id = ?
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (table_index_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_table_id(self, table_id: str) -> dict[str, Any] | None:
        sql = f"""
            SELECT id, raw_file_id, table_id, name, columns, row_count,
                   summary, metadata
            FROM {TABLE} WHERE table_id = ?
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (table_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        sql = f"""
            SELECT id, raw_file_id, table_id, name, columns, row_count,
                   summary, metadata
            FROM {TABLE}
            WHERE raw_file_id = ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_summary(self, table_index_id: str, summary: str) -> None:
        sql = f"UPDATE {TABLE} SET summary = ? WHERE id = ?"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (summary, table_index_id))
            conn.commit()

    def delete_by_file(self, raw_file_id: str) -> None:
        store_utils.delete_by_file(self._cm, self._collection, TABLE, raw_file_id)


def _row_to_dict(row: tuple) -> dict[str, Any]:
    columns = store_utils.decode_json(row[4], [])
    if not isinstance(columns, list):
        columns = []
    meta = store_utils.decode_json(row[7], {})
    if not isinstance(meta, dict):
        meta = {}
    return {
        "id": row[0],
        "raw_file_id": row[1],
        "table_id": row[2],
        "name": row[3],
        "columns": list(columns),
        "row_count": row[5],
        "summary": row[6],
        "metadata": meta,
    }
