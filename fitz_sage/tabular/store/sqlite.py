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

from fitz_sage.core.identifiers import contains_exact_identifier
from fitz_sage.engines.fitz_krag.ingestion.store_utils import build_fts_query
from fitz_sage.logging.logger import get_logger
from fitz_sage.logging.tags import STORAGE
from fitz_sage.storage import get_connection_manager
from fitz_sage.tabular.store.base import StoredTable, compute_hash

logger = get_logger(__name__)

_MAX_ROW_SEARCH_TEXTS = 5
_ROW_SEARCH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


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
    ROW_FTS_TABLE = "_table_row_fts"
    ROW_FTS_SQL = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {ROW_FTS_TABLE} USING fts5(
            table_id UNINDEXED,
            row_num UNINDEXED,
            content,
            tokenize = 'unicode61'
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
            conn.execute(self.ROW_FTS_SQL)
            self._backfill_row_index(conn)
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
            self._replace_row_index(conn, table_id, rows)

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

    def catalog(self) -> list[dict[str, Any]]:
        """Return metadata for every stored table."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            rows = conn.execute(
                """
                SELECT table_id, table_name, columns, column_names_original,
                       row_count, source_file
                FROM _table_metadata
                ORDER BY source_file, table_id
                """
            ).fetchall()

        catalog: list[dict[str, Any]] = []
        for row in rows:
            sanitized = json.loads(row[2]) if row[2] else []
            original = json.loads(row[3]) if row[3] else []
            catalog.append(
                {
                    "table_id": row[0],
                    "table_name": row[1],
                    "columns": list(sanitized),
                    "original_columns": list(original),
                    "row_count": row[4],
                    "source_file": row[5] or "",
                }
            )
        return catalog

    def scan_rows(
        self,
        table_id: str,
        *,
        limit: int = 500,
    ) -> tuple[list[str], list[list[Any]]] | None:
        """Return a bounded row scan for deterministic retrieval planning."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                """
                SELECT table_name, columns, column_names_original
                FROM _table_metadata
                WHERE table_id = ?
                """,
                (table_id,),
            ).fetchone()
            if not result:
                return None
            table_name, sanitized_json, original_json = result
            sanitized_cols = json.loads(sanitized_json) if sanitized_json else []
            original_cols = json.loads(original_json) if original_json else []
            if not sanitized_cols:
                return list(original_cols), []
            cols_str = ", ".join(f'"{column}"' for column in sanitized_cols)
            bounded_limit = max(1, int(limit))
            rows = conn.execute(
                f'SELECT {cols_str} FROM "{table_name}" ORDER BY _row_num LIMIT ?',
                (bounded_limit,),
            ).fetchall()
        return list(original_cols), [list(row) for row in rows]

    def get_rows_by_numbers(
        self,
        table_id: str,
        row_numbers: list[int] | tuple[int, ...],
        *,
        limit: int = 20,
    ) -> tuple[list[str], list[list[Any]]] | None:
        """Return specific source rows while preserving their requested order."""
        requested = list(dict.fromkeys(int(value) for value in row_numbers))
        if not requested:
            return None
        requested = requested[: max(1, int(limit))]

        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                """
                SELECT table_name, columns, column_names_original
                FROM _table_metadata
                WHERE table_id = ?
                """,
                (table_id,),
            ).fetchone()
            if not result:
                return None

            table_name, sanitized_json, original_json = result
            sanitized_cols = json.loads(sanitized_json) if sanitized_json else []
            original_cols = json.loads(original_json) if original_json else []
            if not sanitized_cols:
                return list(original_cols), []

            cols_str = ", ".join(f'"{column}"' for column in sanitized_cols)
            placeholders = ", ".join("?" for _ in requested)
            rows = conn.execute(
                f'SELECT _row_num, {cols_str} FROM "{table_name}" '
                f"WHERE _row_num IN ({placeholders})",
                tuple(requested),
            ).fetchall()

        by_number = {int(row[0]): list(row[1:]) for row in rows}
        return list(original_cols), [
            by_number[row_number] for row_number in requested if row_number in by_number
        ]

    def search_rows_bm25(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return table identities ranked by BM25 matches over concrete row values."""
        fts_query = build_fts_query(query)
        if fts_query is None:
            return []
        query_terms = _meaningful_row_terms(query)
        self._ensure_schema()
        hit_limit = max(limit * 20, 100)
        with self._manager.connection(self.collection) as conn:
            rows = conn.execute(
                f"""
                SELECT table_id, row_num, bm25({self.ROW_FTS_TABLE}) AS rank, content
                FROM {self.ROW_FTS_TABLE}
                WHERE {self.ROW_FTS_TABLE} MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, hit_limit),
            ).fetchall()

        by_table: dict[str, dict[str, Any]] = {}
        for table_id, row_num, rank, content in rows:
            key = str(table_id)
            content_terms = set(_meaningful_row_terms(str(content or "")))
            matched_terms = [term for term in query_terms if term in content_terms]
            coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
            entry = by_table.get(key)
            if entry is None:
                if len(by_table) >= limit:
                    continue
                entry = {
                    "table_id": key,
                    "rank": len(by_table) + 1,
                    "bm25_score": -float(rank) if rank is not None else 0.0,
                    "matched_rows": 0,
                    "row_numbers": [],
                    "row_texts": [],
                    "query_terms": list(query_terms),
                    "matched_terms": [],
                    "term_coverage": 0.0,
                }
                by_table[key] = entry
            entry["matched_rows"] += 1
            if len(entry["row_numbers"]) < 20:
                entry["row_numbers"].append(int(row_num))
            if content and len(entry["row_texts"]) < _MAX_ROW_SEARCH_TEXTS:
                entry["row_texts"].append(str(content))
            if coverage > float(entry["term_coverage"]):
                entry["matched_terms"] = matched_terms
                entry["term_coverage"] = coverage
        return list(by_table.values())

    def find_rows_by_identifiers(
        self,
        table_id: str,
        identifiers: list[str] | tuple[str, ...],
        *,
        limit: int = 100,
    ) -> tuple[list[str], list[list[Any]]] | None:
        """Find rows containing complete literal identifiers in any column."""
        literal_identifiers = tuple(value for value in identifiers if value)
        if not literal_identifiers:
            return None

        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                """
                SELECT table_name, columns, column_names_original
                FROM _table_metadata
                WHERE table_id = ?
                """,
                (table_id,),
            ).fetchone()
            if not result:
                return None

            table_name, sanitized_json, original_json = result
            sanitized_cols = json.loads(sanitized_json) if sanitized_json else []
            original_cols = json.loads(original_json) if original_json else []
            if not sanitized_cols:
                return list(original_cols), []

            def contains_any_identifier(value: Any) -> int:
                text = "" if value is None else str(value)
                return int(
                    any(
                        contains_exact_identifier(text, identifier)
                        for identifier in literal_identifiers
                    )
                )

            conn.create_function(
                "fitz_contains_exact_identifier",
                1,
                contains_any_identifier,
                deterministic=True,
            )
            cols_str = ", ".join(f'"{column}"' for column in sanitized_cols)
            predicates = " OR ".join(
                f'fitz_contains_exact_identifier("{column}") = 1' for column in sanitized_cols
            )
            bounded_limit = max(1, int(limit))
            rows = conn.execute(
                f'SELECT {cols_str} FROM "{table_name}" '
                f"WHERE {predicates} ORDER BY _row_num LIMIT ?",
                (bounded_limit,),
            ).fetchall()

        return list(original_cols), [list(row) for row in rows]

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
                self._rebuild_row_index(
                    conn,
                    table_id,
                    table_name,
                    updated_cols,
                    updated_original,
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
                conn.execute(
                    f"DELETE FROM {self.ROW_FTS_TABLE} WHERE table_id = ?",
                    (table_id,),
                )
                conn.execute("DELETE FROM _table_metadata WHERE table_id = ?", (table_id,))
                conn.commit()
                logger.debug(f"{STORAGE} Deleted table '{table_id}' ('{table_name}')")

    def close(self) -> None:
        """No-op (no persistent connection)."""
        pass

    def _backfill_row_index(self, conn: Any) -> None:
        """Populate row FTS for collections created before the row index existed."""
        missing = conn.execute(
            f"""
            SELECT m.table_id, m.table_name, m.columns, m.column_names_original
            FROM _table_metadata m
            LEFT JOIN (
                SELECT DISTINCT table_id FROM {self.ROW_FTS_TABLE}
            ) row_index ON row_index.table_id = m.table_id
            WHERE row_index.table_id IS NULL
            """
        ).fetchall()
        for table_id, table_name, columns_json, original_json in missing:
            columns = json.loads(columns_json) if columns_json else []
            original = json.loads(original_json) if original_json else []
            self._rebuild_row_index(
                conn,
                str(table_id),
                str(table_name),
                list(columns),
                list(original),
            )

    def _rebuild_row_index(
        self,
        conn: Any,
        table_id: str,
        table_name: str,
        sanitized_columns: list[str],
        original_columns: list[str],
    ) -> None:
        conn.execute(
            f"DELETE FROM {self.ROW_FTS_TABLE} WHERE table_id = ?",
            (table_id,),
        )
        if not sanitized_columns:
            return
        columns_sql = ", ".join(f'"{column}"' for column in sanitized_columns)
        rows = conn.execute(
            f'SELECT _row_num, {columns_sql} FROM "{table_name}" ORDER BY _row_num'
        ).fetchall()
        index_rows = [
            (
                table_id,
                int(row[0]),
                _row_index_text(list(row[1:])),
            )
            for row in rows
        ]
        if index_rows:
            conn.executemany(
                f"""
                INSERT INTO {self.ROW_FTS_TABLE} (table_id, row_num, content)
                VALUES (?, ?, ?)
                """,
                index_rows,
            )

    def _replace_row_index(
        self,
        conn: Any,
        table_id: str,
        rows: list[list[str]],
    ) -> None:
        conn.execute(
            f"DELETE FROM {self.ROW_FTS_TABLE} WHERE table_id = ?",
            (table_id,),
        )
        index_rows = [(table_id, row_num, _row_index_text(row)) for row_num, row in enumerate(rows)]
        if index_rows:
            conn.executemany(
                f"""
                INSERT INTO {self.ROW_FTS_TABLE} (table_id, row_num, content)
                VALUES (?, ?, ?)
                """,
                index_rows,
            )


def _row_index_text(row: list[Any]) -> str:
    return " ".join(str(value) for value in row if value is not None)


def _meaningful_row_terms(value: str) -> tuple[str, ...]:
    """Return stable lexical terms used to qualify row-index matches."""
    terms = [
        term.casefold()
        for term in re.findall(r"[^\W_]+", value, flags=re.UNICODE)
        if term.casefold() not in _ROW_SEARCH_STOPWORDS
    ]
    return tuple(dict.fromkeys(terms))


__all__ = ["SqliteTableStore"]
