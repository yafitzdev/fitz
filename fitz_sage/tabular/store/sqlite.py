# fitz_sage/tabular/store/sqlite.py
"""
SQLite-based table store with native table support.

Each tabular file ingested becomes a real SQLite table (one per
``table_id``) that SQL queries can hit directly without loading the
rows into memory. Metadata about every stored table lives in the
``_table_metadata`` table.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fitz_sage.logging.logger import get_logger
from fitz_sage.logging.tags import STORAGE
from fitz_sage.storage import get_connection_manager
from fitz_sage.tabular.store.base import StoredTable, compute_hash

logger = get_logger(__name__)


def _sanitize_table_name(table_id: str) -> str:
    """Convert table_id to a valid SQLite table name."""
    name = re.sub(r"[^a-zA-Z0-9]", "_", table_id)
    if name and name[0].isdigit():
        name = "t_" + name
    return f"tbl_{name[:55]}".lower()


def _sanitize_column_name(col: str) -> str:
    """Convert column name to a valid SQLite identifier."""
    name = re.sub(r"[^a-zA-Z0-9]", "_", col)
    if name and name[0].isdigit():
        name = "c_" + name
    if not name:
        name = "col"
    return name.lower()


class SqliteTableStore:
    """Table storage using native SQLite tables.

    Schema:
    - ``_table_metadata``: registry of all tables (id, hash, columns, source)
    - ``tbl_{sanitized_id}``: actual data tables with columns as TEXT
    """

    METADATA_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS _table_metadata (
            table_id TEXT PRIMARY KEY,
            table_name TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            columns TEXT NOT NULL,
            column_names_original TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            source_file TEXT,
            file_hash TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """

    def __init__(self, collection: str):
        self.collection = collection
        self._manager = get_connection_manager()
        self._manager.start()
        self._schema_initialized = False

    def _ensure_schema(self) -> None:
        if self._schema_initialized:
            return

        with self._manager.connection(self.collection) as conn:
            conn.execute(self.METADATA_TABLE_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug(f"{STORAGE} Table metadata schema initialized for '{self.collection}'")

    def store(
        self,
        table_id: str,
        columns: list[str],
        rows: list[list[str]],
        source_file: str,
        file_hash: str | None = None,
    ) -> str:
        self._ensure_schema()

        content_hash = compute_hash(columns, rows)
        table_name = _sanitize_table_name(table_id)
        sanitized_cols = [_sanitize_column_name(c) for c in columns]

        seen: dict[str, int] = {}
        unique_cols = []
        for col in sanitized_cols:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        sanitized_cols = unique_cols

        with self._manager.connection(self.collection) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

            cols_def = "_row_num INTEGER PRIMARY KEY, " + ", ".join(
                f'"{c}" TEXT' for c in sanitized_cols
            )
            conn.execute(f'CREATE TABLE "{table_name}" ({cols_def})')

            if rows:
                placeholders = ", ".join(["?"] * (len(sanitized_cols) + 1))
                insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                batch_size = 1000
                for batch_start in range(0, len(rows), batch_size):
                    batch = rows[batch_start : batch_start + batch_size]
                    padded_batch = []
                    for i, row in enumerate(batch):
                        row_num = batch_start + i
                        if len(row) < len(sanitized_cols):
                            row = row + [""] * (len(sanitized_cols) - len(row))
                        elif len(row) > len(sanitized_cols):
                            row = row[: len(sanitized_cols)]
                        padded_batch.append((row_num, *(None if v == "" else v for v in row)))
                    conn.executemany(insert_sql, padded_batch)

            conn.execute(
                """
                INSERT INTO _table_metadata
                    (table_id, table_name, hash, columns, column_names_original,
                     row_count, source_file, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(table_id) DO UPDATE SET
                    table_name = excluded.table_name,
                    hash = excluded.hash,
                    columns = excluded.columns,
                    column_names_original = excluded.column_names_original,
                    row_count = excluded.row_count,
                    source_file = excluded.source_file,
                    file_hash = excluded.file_hash
                """,
                (
                    table_id,
                    table_name,
                    content_hash,
                    json.dumps(sanitized_cols),
                    json.dumps(list(columns)),
                    len(rows),
                    source_file,
                    file_hash,
                ),
            )
            conn.commit()

        logger.debug(f"{STORAGE} Stored table '{table_id}' as '{table_name}' ({len(rows)} rows)")
        return content_hash

    def retrieve(self, table_id: str) -> StoredTable | None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                """
                SELECT table_name, hash, columns, column_names_original, row_count, source_file
                FROM _table_metadata
                WHERE table_id = ?
                """,
                (table_id,),
            ).fetchone()

            if not result:
                return None

            table_name, hash_, columns_json, original_json, row_count, source_file = result
            columns = json.loads(columns_json) if columns_json else []
            original_columns = json.loads(original_json) if original_json else []

            try:
                cols_str = ", ".join(f'"{c}"' for c in columns)
                cursor = conn.execute(f'SELECT {cols_str} FROM "{table_name}" ORDER BY _row_num')
                rows = [list(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.warning(f"{STORAGE} Failed to fetch data from '{table_name}': {e}")
                rows = []

            return StoredTable(
                table_id=table_id,
                hash=hash_,
                columns=list(original_columns),
                rows=rows,
                row_count=row_count,
                source_file=source_file or "",
            )

    def get_table_name(self, table_id: str) -> str | None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT table_name FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def get_columns(self, table_id: str) -> tuple[list[str], list[str]] | None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT columns, column_names_original FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            if result:
                cols = json.loads(result[0]) if result[0] else []
                orig = json.loads(result[1]) if result[1] else []
                return list(cols), list(orig)
            return None

    def execute_query(
        self,
        table_id: str,
        sql: str,
        params: tuple = (),
    ) -> tuple[list[str], list[list[Any]]] | None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            try:
                cursor = conn.execute(sql, params)
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [list(row) for row in cursor.fetchall()]
                return col_names, rows
            except Exception as e:
                logger.warning(f"{STORAGE} Query execution failed: {e}")
                return None

    def execute_multi_table_query(
        self,
        sql: str,
        params: tuple = (),
    ) -> tuple[list[str], list[list[Any]]] | None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            try:
                cursor = conn.execute(sql, params)
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [list(row) for row in cursor.fetchall()]
                return col_names, rows
            except Exception as e:
                logger.warning(f"{STORAGE} Multi-table query failed: {e}")
                return None

    def get_hash(self, table_id: str) -> str | None:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT hash FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def get_row_count(self, table_id: str) -> int | None:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT row_count FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def get_file_hash(self, table_id: str) -> str | None:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT file_hash FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()
            return result[0] if result else None

    def add_columns(
        self,
        table_id: str,
        new_columns: list[str],
        column_values: list[list[str]],
    ) -> bool:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                """
                SELECT table_name, columns, column_names_original
                FROM _table_metadata WHERE table_id = ?
                """,
                (table_id,),
            ).fetchone()

            if not result:
                logger.warning(f"{STORAGE} Table '{table_id}' not found for column addition")
                return False

            table_name, existing_cols_json, existing_original_json = result
            existing_cols = json.loads(existing_cols_json) if existing_cols_json else []
            existing_original = json.loads(existing_original_json) if existing_original_json else []

            sanitized_new = [_sanitize_column_name(c) for c in new_columns]
            for i, col in enumerate(sanitized_new):
                if col in existing_cols:
                    suffix = 1
                    while f"{col}_{suffix}" in existing_cols:
                        suffix += 1
                    sanitized_new[i] = f"{col}_{suffix}"

            try:
                for col in sanitized_new:
                    conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN "{col}" TEXT')

                for row_num, row_values in enumerate(column_values):
                    if not row_values:
                        continue
                    set_clause = ", ".join(
                        f'"{col}" = ?' for col in sanitized_new[: len(row_values)]
                    )
                    conn.execute(
                        f'UPDATE "{table_name}" SET {set_clause} WHERE _row_num = ?',
                        (*row_values[: len(sanitized_new)], row_num),
                    )

                updated_cols = existing_cols + sanitized_new
                updated_original = existing_original + new_columns
                conn.execute(
                    """
                    UPDATE _table_metadata
                    SET columns = ?, column_names_original = ?
                    WHERE table_id = ?
                    """,
                    (json.dumps(updated_cols), json.dumps(updated_original), table_id),
                )
                conn.commit()

                logger.debug(
                    f"{STORAGE} Added {len(new_columns)} columns to '{table_id}': {new_columns}"
                )
                return True

            except Exception as e:
                logger.warning(f"{STORAGE} Failed to add columns to '{table_id}': {e}")
                conn.rollback()
                return False

    def has_columns(self, table_id: str, columns: list[str]) -> tuple[list[str], list[str]]:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT column_names_original FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()

            if not result:
                return [], columns

            existing_original = set(json.loads(result[0]) if result[0] else [])
            existing = [c for c in columns if c in existing_original]
            missing = [c for c in columns if c not in existing_original]
            return existing, missing

    def list_tables(self) -> list[str]:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute("SELECT table_id FROM _table_metadata ORDER BY table_id")
            return [row[0] for row in cursor]

    def delete(self, table_id: str) -> None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT table_name FROM _table_metadata WHERE table_id = ?",
                (table_id,),
            ).fetchone()

            if result:
                table_name = result[0]
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.execute("DELETE FROM _table_metadata WHERE table_id = ?", (table_id,))
                conn.commit()
                logger.debug(f"{STORAGE} Deleted table '{table_id}' ('{table_name}')")

    def close(self) -> None:
        """No-op (no persistent connection)."""
        pass


__all__ = ["SqliteTableStore"]
