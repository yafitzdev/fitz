# fitz_sage/engines/fitz_krag/ingestion/symbol_store.py
"""CRUD operations for krag_symbol_index table."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}symbol_index"


class SymbolStore:
    """CRUD for the symbol index."""

    def __init__(self, connection_manager: "PostgresConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, symbols: list[dict[str, Any]]) -> None:
        """Insert or update a batch of symbols."""
        if not symbols:
            return

        sql = f"""
            INSERT INTO {TABLE}
                ("id", "name", "qualified_name", "kind", "raw_file_id",
                 "start_line", "end_line", "signature", "summary",
                 "imports", "references", "keywords", "entities", "metadata")
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s, %s, %s,
                 %s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT ("id") DO UPDATE SET
                "name" = EXCLUDED."name",
                "qualified_name" = EXCLUDED."qualified_name",
                "kind" = EXCLUDED."kind",
                "start_line" = EXCLUDED."start_line",
                "end_line" = EXCLUDED."end_line",
                "signature" = EXCLUDED."signature",
                "summary" = EXCLUDED."summary",
                "imports" = EXCLUDED."imports",
                "references" = EXCLUDED."references",
                "keywords" = EXCLUDED."keywords",
                "entities" = EXCLUDED."entities",
                "metadata" = EXCLUDED."metadata"
        """
        with self._cm.connection(self._collection) as conn:
            for sym in symbols:
                conn.execute(
                    sql,
                    (
                        sym["id"],
                        sym["name"],
                        sym["qualified_name"],
                        sym["kind"],
                        sym["raw_file_id"],
                        sym["start_line"],
                        sym["end_line"],
                        sym.get("signature"),
                        sym.get("summary"),
                        sym.get("imports", []),
                        sym.get("references", []),
                        sym.get("keywords", []),
                        json.dumps(sym.get("entities", [])),
                        json.dumps(sym.get("metadata", {})),
                    ),
                )
            conn.commit()

    def search_bm25(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search on symbol name + summary using tsvector."""
        sql = f"""
            SELECT "id", "name", "qualified_name", "kind", "raw_file_id",
                   "start_line", "end_line", "signature", "summary", "metadata",
                   ts_rank_cd("content_tsv", plainto_tsquery('english', %s)) AS "rank"
            FROM {TABLE}
            WHERE "content_tsv" @@ plainto_tsquery('english', %s)
            ORDER BY "rank" DESC
            LIMIT %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (query, query, limit)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:10])
            d["bm25_score"] = float(row[10]) if row[10] is not None else 0.0
            results.append(d)
        return results

    def search_by_name(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Keyword search against symbol name and qualified_name using ILIKE."""
        pattern = f"%{query}%"
        sql = f"""
            SELECT "id", "name", "qualified_name", "kind", "raw_file_id",
                   "start_line", "end_line", "signature", "summary", "metadata"
            FROM {TABLE}
            WHERE "name" ILIKE %s OR "qualified_name" ILIKE %s
            LIMIT %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (pattern, pattern, limit)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def delete_by_file(self, raw_file_id: str) -> None:
        """Delete all symbols for a raw file."""
        sql = f'DELETE FROM {TABLE} WHERE "raw_file_id" = %s'
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (raw_file_id,))
            conn.commit()

    def get(self, symbol_id: str) -> dict[str, Any] | None:
        """Get a symbol by ID."""
        sql = f"""
            SELECT "id", "name", "qualified_name", "kind", "raw_file_id",
                   "start_line", "end_line", "signature", "summary", "metadata"
            FROM {TABLE} WHERE "id" = %s
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (symbol_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        """Get all symbols for a raw file, ordered by start_line.

        Includes the ``references`` column (index 10) as a separate field
        so that callers can see AST-extracted references without breaking
        existing ``_row_to_dict`` consumers.
        """
        sql = f"""
            SELECT "id", "name", "qualified_name", "kind", "raw_file_id",
                   "start_line", "end_line", "signature", "summary", "metadata",
                   "references"
            FROM {TABLE}
            WHERE "raw_file_id" = %s
            ORDER BY "start_line"
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:10])
            d["references"] = list(row[10]) if row[10] else []
            results.append(d)
        return results

    def search_by_keywords(self, terms: list[str], limit: int = 20) -> list[dict[str, Any]]:
        """Find symbols with matching enriched keywords (array overlap)."""
        if not terms:
            return []
        sql = f"""
            SELECT "id", "name", "qualified_name", "kind", "raw_file_id",
                   "start_line", "end_line", "signature", "summary", "metadata"
            FROM {TABLE}
            WHERE "keywords" && %s
            LIMIT %s
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (terms, limit)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_structural_manifest(self) -> list[dict]:
        """Get a compact structural manifest of all symbols grouped by file.

        Returns a list of per-file dicts:
        {
            "raw_file_id": str,
            "path": str,
            "symbols": [{"name", "qualified_name", "kind", "signature",
                         "start_line", "end_line", "imports"}]
        }
        """
        sql = f"""
            SELECT s."raw_file_id", r."path",
                   s."name", s."qualified_name", s."kind", s."signature",
                   s."start_line", s."end_line", s."imports"
            FROM {TABLE} s
            JOIN {TABLE_PREFIX}raw_files r ON s."raw_file_id" = r."id"
            ORDER BY r."path", s."start_line"
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()

        files: dict[str, dict] = {}
        for row in rows:
            fid = row[0]
            if fid not in files:
                files[fid] = {"raw_file_id": fid, "path": row[1], "symbols": []}
            files[fid]["symbols"].append(
                {
                    "name": row[2],
                    "qualified_name": row[3],
                    "kind": row[4],
                    "signature": row[5],
                    "start_line": row[6],
                    "end_line": row[7],
                    "imports": list(row[8]) if row[8] else [],
                }
            )
        return list(files.values())

    def get_summaries_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        """Get symbol IDs, names, kinds, and summaries for a file."""
        sql = f"""
            SELECT "id", "name", "kind", "summary"
            FROM {TABLE}
            WHERE "raw_file_id" = %s
            ORDER BY "start_line"
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        return [{"id": row[0], "name": row[1], "kind": row[2], "summary": row[3]} for row in rows]

    def update_summaries_by_file(self, raw_file_id: str, summaries: list[str]) -> None:
        """Update summaries for all symbols in a file, in start_line order."""
        ids_sql = f"""
            SELECT "id" FROM {TABLE}
            WHERE "raw_file_id" = %s
            ORDER BY "start_line"
        """
        update_sql = f'UPDATE {TABLE} SET "summary" = %s WHERE "id" = %s'
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(ids_sql, (raw_file_id,)).fetchall()
            for i, row in enumerate(rows):
                if i < len(summaries):
                    conn.execute(update_sql, (summaries[i], row[0]))
            conn.commit()

    def update_enrichment_by_file(
        self, raw_file_id: str, enriched_dicts: list[dict[str, Any]]
    ) -> None:
        """Update keywords, entities, and metadata for symbols in a file."""
        sql = (
            f'UPDATE {TABLE} SET "keywords" = %s, "entities" = %s::jsonb,'
            f' "metadata" = %s::jsonb WHERE "id" = %s'
        )
        with self._cm.connection(self._collection) as conn:
            for item in enriched_dicts:
                conn.execute(
                    sql,
                    (
                        item.get("keywords", []),
                        json.dumps(item.get("entities", [])),
                        json.dumps(item.get("metadata", {})),
                        item["id"],
                    ),
                )
            conn.commit()

def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a query row to dict."""
    meta = row[9]
    if isinstance(meta, str):
        meta = json.loads(meta)
    elif meta is None:
        meta = {}
    return {
        "id": row[0],
        "name": row[1],
        "qualified_name": row[2],
        "kind": row[3],
        "raw_file_id": row[4],
        "start_line": row[5],
        "end_line": row[6],
        "signature": row[7],
        "summary": row[8],
        "metadata": meta,
    }
