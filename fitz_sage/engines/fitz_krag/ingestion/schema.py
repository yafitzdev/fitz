# fitz_sage/engines/fitz_krag/ingestion/schema.py
"""
Database schema for Fitz KRAG engine on SQLite.

Tables:
- krag_raw_files: stores original file content (keyed by content hash)
- krag_symbol_index: code symbol registry (+ FTS5 over name/qualified_name/summary)
- krag_import_graph: file-level dependency links
- krag_section_index: document section registry (+ FTS5 over title/content)
- krag_table_index: table metadata registry (+ FTS5 over name)

Full-text search uses SQLite FTS5 with external-content tables and
triggers that keep the FTS index in sync with the base tables on
INSERT / UPDATE / DELETE. Ranking is via the built-in ``bm25()``
function (lower = more relevant). Arrays (TEXT[] in the Postgres
schema) and JSONB columns are stored as TEXT containing JSON; query
sites use ``json_each`` / ``json_extract`` for traversal.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager

logger = logging.getLogger(__name__)

TABLE_PREFIX = "krag_"


def _raw_files_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}raw_files (
        id TEXT PRIMARY KEY,
        path TEXT NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        file_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{{}}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}raw_files_path
        ON {TABLE_PREFIX}raw_files(path);
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}raw_files_hash
        ON {TABLE_PREFIX}raw_files(content_hash);
    """


def _symbol_index_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}symbol_index (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        qualified_name TEXT NOT NULL,
        kind TEXT NOT NULL,
        raw_file_id TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files(id) ON DELETE CASCADE,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        signature TEXT,
        summary TEXT,
        imports TEXT NOT NULL DEFAULT '[]',
        "references" TEXT NOT NULL DEFAULT '[]',
        keywords TEXT NOT NULL DEFAULT '[]',
        entities TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{{}}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_name
        ON {TABLE_PREFIX}symbol_index(name);
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_qualified
        ON {TABLE_PREFIX}symbol_index(qualified_name);
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_kind
        ON {TABLE_PREFIX}symbol_index(kind);
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_file
        ON {TABLE_PREFIX}symbol_index(raw_file_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_PREFIX}symbol_fts USING fts5(
        name, qualified_name, summary,
        content='{TABLE_PREFIX}symbol_index',
        content_rowid='rowid',
        tokenize='porter unicode61'
    );
    CREATE TRIGGER IF NOT EXISTS {TABLE_PREFIX}symbol_ai
        AFTER INSERT ON {TABLE_PREFIX}symbol_index BEGIN
        INSERT INTO {TABLE_PREFIX}symbol_fts(rowid, name, qualified_name, summary)
        VALUES (new.rowid, new.name, new.qualified_name, COALESCE(new.summary, ''));
    END;
    CREATE TRIGGER IF NOT EXISTS {TABLE_PREFIX}symbol_ad
        AFTER DELETE ON {TABLE_PREFIX}symbol_index BEGIN
        INSERT INTO {TABLE_PREFIX}symbol_fts({TABLE_PREFIX}symbol_fts, rowid, name, qualified_name, summary)
        VALUES('delete', old.rowid, old.name, old.qualified_name, COALESCE(old.summary, ''));
    END;
    CREATE TRIGGER IF NOT EXISTS {TABLE_PREFIX}symbol_au
        AFTER UPDATE ON {TABLE_PREFIX}symbol_index BEGIN
        INSERT INTO {TABLE_PREFIX}symbol_fts({TABLE_PREFIX}symbol_fts, rowid, name, qualified_name, summary)
        VALUES('delete', old.rowid, old.name, old.qualified_name, COALESCE(old.summary, ''));
        INSERT INTO {TABLE_PREFIX}symbol_fts(rowid, name, qualified_name, summary)
        VALUES (new.rowid, new.name, new.qualified_name, COALESCE(new.summary, ''));
    END;
    """


def _import_graph_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}import_graph (
        source_file_id TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files(id) ON DELETE CASCADE,
        target_module TEXT NOT NULL,
        target_file_id TEXT REFERENCES {TABLE_PREFIX}raw_files(id) ON DELETE SET NULL,
        import_names TEXT NOT NULL DEFAULT '[]',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (source_file_id, target_module)
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}import_target
        ON {TABLE_PREFIX}import_graph(target_file_id);
    """


def _section_index_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}section_index (
        id TEXT PRIMARY KEY,
        raw_file_id TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        level INTEGER NOT NULL,
        page_start INTEGER,
        page_end INTEGER,
        content TEXT NOT NULL,
        summary TEXT,
        parent_section_id TEXT,
        position INTEGER NOT NULL,
        keywords TEXT NOT NULL DEFAULT '[]',
        entities TEXT NOT NULL DEFAULT '[]',
        metadata TEXT NOT NULL DEFAULT '{{}}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}section_file
        ON {TABLE_PREFIX}section_index(raw_file_id);
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}section_parent
        ON {TABLE_PREFIX}section_index(parent_section_id);

    CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_PREFIX}section_fts USING fts5(
        title, content,
        content='{TABLE_PREFIX}section_index',
        content_rowid='rowid',
        tokenize='porter unicode61'
    );
    CREATE TRIGGER IF NOT EXISTS {TABLE_PREFIX}section_ai
        AFTER INSERT ON {TABLE_PREFIX}section_index BEGIN
        INSERT INTO {TABLE_PREFIX}section_fts(rowid, title, content)
        VALUES (new.rowid, new.title, new.content);
    END;
    CREATE TRIGGER IF NOT EXISTS {TABLE_PREFIX}section_ad
        AFTER DELETE ON {TABLE_PREFIX}section_index BEGIN
        INSERT INTO {TABLE_PREFIX}section_fts({TABLE_PREFIX}section_fts, rowid, title, content)
        VALUES('delete', old.rowid, old.title, old.content);
    END;
    CREATE TRIGGER IF NOT EXISTS {TABLE_PREFIX}section_au
        AFTER UPDATE ON {TABLE_PREFIX}section_index BEGIN
        INSERT INTO {TABLE_PREFIX}section_fts({TABLE_PREFIX}section_fts, rowid, title, content)
        VALUES('delete', old.rowid, old.title, old.content);
        INSERT INTO {TABLE_PREFIX}section_fts(rowid, title, content)
        VALUES (new.rowid, new.title, new.content);
    END;
    """


def _table_index_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}table_index (
        id TEXT PRIMARY KEY,
        raw_file_id TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files(id) ON DELETE CASCADE,
        table_id TEXT NOT NULL,
        name TEXT NOT NULL,
        columns TEXT NOT NULL DEFAULT '[]',
        row_count INTEGER NOT NULL,
        summary TEXT,
        metadata TEXT NOT NULL DEFAULT '{{}}',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}table_table_id
        ON {TABLE_PREFIX}table_index(table_id);
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}table_file
        ON {TABLE_PREFIX}table_index(raw_file_id);
    """


def _exec_script(conn, script: str) -> None:
    """Run a DDL block (multiple statements separated by ;)."""
    conn.executescript(script)


def ensure_schema(
    connection_manager: "SqliteConnectionManager",
    collection: str,
) -> None:
    """Create KRAG tables, FTS5 virtual tables, and sync triggers.

    Called on engine init / first ingest. Safe to call multiple times
    thanks to ``IF NOT EXISTS`` on every object.
    """
    with connection_manager.connection(collection) as conn:
        _exec_script(conn, _raw_files_ddl())
        _exec_script(conn, _symbol_index_ddl())
        _exec_script(conn, _import_graph_ddl())
        _exec_script(conn, _section_index_ddl())
        _exec_script(conn, _table_index_ddl())
        conn.commit()

    logger.info(f"KRAG schema ensured for collection '{collection}'")
