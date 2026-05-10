# fitz_sage/engines/fitz_krag/ingestion/schema.py
"""
Database schema for Fitz KRAG engine.

All column identifiers are double-quoted to avoid reserved-word collisions
(e.g. ``references`` is a PostgreSQL reserved word).

Tables:
- krag_raw_files: stores original file content (keyed by content hash)
- krag_symbol_index: code symbol registry (BM25 over signatures + summaries)
- krag_import_graph: file-level dependency links
- krag_section_index: document section registry (BM25 over title + content)
- krag_table_index: table metadata registry
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_sage.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

# Prefix tables to avoid collision with other engine tables
TABLE_PREFIX = "krag_"


def _raw_files_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}raw_files (
        "id" TEXT PRIMARY KEY,
        "path" TEXT NOT NULL,
        "content" TEXT NOT NULL,
        "content_hash" TEXT NOT NULL,
        "file_type" TEXT NOT NULL,
        "size_bytes" INTEGER NOT NULL,
        "metadata" JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        "created_at" TIMESTAMPTZ DEFAULT NOW(),
        "updated_at" TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}raw_files_path
        ON {TABLE_PREFIX}raw_files ("path");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}raw_files_hash
        ON {TABLE_PREFIX}raw_files ("content_hash");
    """


def _symbol_index_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}symbol_index (
        "id" TEXT PRIMARY KEY,
        "name" TEXT NOT NULL,
        "qualified_name" TEXT NOT NULL,
        "kind" TEXT NOT NULL,
        "raw_file_id" TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files("id") ON DELETE CASCADE,
        "start_line" INTEGER NOT NULL,
        "end_line" INTEGER NOT NULL,
        "signature" TEXT,
        "summary" TEXT,
        "imports" TEXT[],
        "references" TEXT[],
        "keywords" TEXT[] DEFAULT '{{}}',
        "entities" JSONB DEFAULT '[]'::jsonb,
        "content_tsv" tsvector GENERATED ALWAYS AS (
            to_tsvector('english',
                coalesce("name", '') || ' ' ||
                coalesce("qualified_name", '') || ' ' ||
                coalesce("summary", ''))
        ) STORED,
        "metadata" JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        "created_at" TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_name
        ON {TABLE_PREFIX}symbol_index ("name");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_qualified
        ON {TABLE_PREFIX}symbol_index ("qualified_name");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_kind
        ON {TABLE_PREFIX}symbol_index ("kind");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_file
        ON {TABLE_PREFIX}symbol_index ("raw_file_id");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}symbol_tsv
        ON {TABLE_PREFIX}symbol_index USING gin ("content_tsv");
    """


def _import_graph_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}import_graph (
        "source_file_id" TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files("id") ON DELETE CASCADE,
        "target_module" TEXT NOT NULL,
        "target_file_id" TEXT REFERENCES {TABLE_PREFIX}raw_files("id") ON DELETE SET NULL,
        "import_names" TEXT[],
        "created_at" TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY ("source_file_id", "target_module")
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}import_target
        ON {TABLE_PREFIX}import_graph ("target_file_id");
    """


def _section_index_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}section_index (
        "id" TEXT PRIMARY KEY,
        "raw_file_id" TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files("id") ON DELETE CASCADE,
        "title" TEXT NOT NULL,
        "level" INTEGER NOT NULL,
        "page_start" INTEGER,
        "page_end" INTEGER,
        "content" TEXT NOT NULL,
        "content_tsv" tsvector GENERATED ALWAYS AS (
            to_tsvector('english', "title" || ' ' || "content")
        ) STORED,
        "summary" TEXT,
        "parent_section_id" TEXT,
        "position" INTEGER NOT NULL,
        "keywords" TEXT[] DEFAULT '{{}}',
        "entities" JSONB DEFAULT '[]'::jsonb,
        "metadata" JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        "created_at" TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}section_title
        ON {TABLE_PREFIX}section_index USING gin (to_tsvector('english', "title"));
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}section_tsv
        ON {TABLE_PREFIX}section_index USING gin ("content_tsv");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}section_file
        ON {TABLE_PREFIX}section_index ("raw_file_id");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}section_parent
        ON {TABLE_PREFIX}section_index ("parent_section_id");
    """


def _table_index_ddl() -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE_PREFIX}table_index (
        "id" TEXT PRIMARY KEY,
        "raw_file_id" TEXT NOT NULL REFERENCES {TABLE_PREFIX}raw_files("id") ON DELETE CASCADE,
        "table_id" TEXT NOT NULL,
        "name" TEXT NOT NULL,
        "columns" TEXT[] NOT NULL,
        "row_count" INTEGER NOT NULL,
        "summary" TEXT,
        "metadata" JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        "created_at" TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}table_name
        ON {TABLE_PREFIX}table_index USING gin (to_tsvector('english', "name"));
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}table_table_id
        ON {TABLE_PREFIX}table_index ("table_id");
    CREATE INDEX IF NOT EXISTS idx_{TABLE_PREFIX}table_file
        ON {TABLE_PREFIX}table_index ("raw_file_id");
    """


def ensure_schema(
    connection_manager: "PostgresConnectionManager",
    collection: str,
) -> None:
    """
    Create KRAG tables if they don't exist.

    Called on engine init / first ingest. Safe to call multiple times.
    """
    with connection_manager.connection(collection) as conn:
        conn.execute(_raw_files_ddl())
        conn.execute(_symbol_index_ddl())
        conn.execute(_import_graph_ddl())
        conn.execute(_section_index_ddl())
        conn.execute(_table_index_ddl())
        conn.commit()

    logger.info(f"KRAG schema ensured for collection '{collection}'")
