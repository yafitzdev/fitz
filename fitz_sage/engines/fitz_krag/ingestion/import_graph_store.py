# fitz_sage/engines/fitz_krag/ingestion/import_graph_store.py
"""CRUD operations for krag_import_graph table."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.code_utils import path_to_module
from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}import_graph"


class ImportGraphStore:
    """CRUD for the import graph."""

    def __init__(self, connection_manager: "SqliteConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return

        sql = f"""
            INSERT INTO {TABLE}
                (source_file_id, target_module, target_file_id, import_names)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_file_id, target_module) DO UPDATE SET
                target_file_id = excluded.target_file_id,
                import_names = excluded.import_names
        """
        with self._cm.connection(self._collection) as conn:
            for edge in edges:
                conn.execute(
                    sql,
                    (
                        edge["source_file_id"],
                        edge["target_module"],
                        edge.get("target_file_id"),
                        json.dumps(edge.get("import_names", [])),
                    ),
                )
            conn.commit()

    def get_imports(self, file_id: str) -> list[dict[str, Any]]:
        sql = f"""
            SELECT source_file_id, target_module, target_file_id, import_names
            FROM {TABLE} WHERE source_file_id = ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (file_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_importers(self, file_id: str) -> list[dict[str, Any]]:
        sql = f"""
            SELECT source_file_id, target_module, target_file_id, import_names
            FROM {TABLE} WHERE target_file_id = ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (file_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_reverse_counts(self) -> dict[str, int]:
        sql = f"""
            SELECT target_file_id, COUNT(*)
            FROM {TABLE}
            WHERE target_file_id IS NOT NULL
            GROUP BY target_file_id
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def delete_by_file(self, file_id: str) -> None:
        sql = f"DELETE FROM {TABLE} WHERE source_file_id = ?"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (file_id,))
            conn.commit()

    def resolve_targets(self, path_to_id: dict[str, str]) -> int:
        """Resolve target_file_id for edges whose target_file_id is NULL."""
        module_to_id: dict[str, str] = {}
        for path, file_id in path_to_id.items():
            module = path_to_module(path, (".py",), "/__init__")
            module_to_id[module] = file_id

        select_sql = f"""
            SELECT source_file_id, target_module
            FROM {TABLE}
            WHERE target_file_id IS NULL
        """
        update_sql = f"""
            UPDATE {TABLE}
            SET target_file_id = ?
            WHERE source_file_id = ? AND target_module = ?
        """
        resolved = 0
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(select_sql).fetchall()
            for source_file_id, target_module in rows:
                target_id = module_to_id.get(target_module)
                if target_id:
                    conn.execute(update_sql, (target_id, source_file_id, target_module))
                    resolved += 1
            if resolved:
                conn.commit()
        return resolved


def _row_to_dict(row: tuple) -> dict[str, Any]:
    names = row[3]
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except (json.JSONDecodeError, TypeError):
            names = []
    elif names is None:
        names = []
    if not isinstance(names, list):
        names = []
    return {
        "source_file_id": row[0],
        "target_module": row[1],
        "target_file_id": row[2],
        "import_names": names,
    }
