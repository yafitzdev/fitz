# fitz_sage/engines/fitz_krag/ingestion/raw_file_store.py
"""CRUD operations for krag_raw_files table."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}raw_files"


class RawFileStore:
    """CRUD for raw file storage."""

    def __init__(self, connection_manager: "SqliteConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert(
        self,
        file_id: str,
        path: str,
        content: str,
        content_hash: str,
        file_type: str,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta_json = json.dumps(metadata or {})
        sql = f"""
            INSERT INTO {TABLE}
                (id, path, content, content_hash, file_type, size_bytes, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                path = excluded.path,
                content = excluded.content,
                content_hash = excluded.content_hash,
                file_type = excluded.file_type,
                size_bytes = excluded.size_bytes,
                metadata = excluded.metadata
        """
        with self._cm.connection(self._collection) as conn:
            conn.execute(
                sql, (file_id, path, content, content_hash, file_type, size_bytes, meta_json)
            )
            conn.commit()

    def get(self, file_id: str) -> dict[str, Any] | None:
        sql = f"""
            SELECT id, path, content, content_hash, file_type, size_bytes, metadata
            FROM {TABLE} WHERE id = ?
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (file_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def delete(self, file_id: str) -> None:
        """Delete a raw file (cascades to symbols + imports via FK)."""
        sql = f"DELETE FROM {TABLE} WHERE id = ?"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (file_id,))
            conn.commit()

    def list_hashes(self) -> dict[str, str]:
        sql = f"SELECT path, content_hash FROM {TABLE}"
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0]: row[1] for row in rows}

    def list_ids_by_path(self) -> dict[str, str]:
        sql = f"SELECT path, id FROM {TABLE}"
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0]: row[1] for row in rows}


def _row_to_dict(row: tuple) -> dict[str, Any]:
    meta = row[6]
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    elif meta is None:
        meta = {}
    return {
        "id": row[0],
        "path": row[1],
        "content": row[2],
        "content_hash": row[3],
        "file_type": row[4],
        "size_bytes": row[5],
        "metadata": meta,
    }
