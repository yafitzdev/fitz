# fitz_sage/engines/fitz_krag/ingestion/symbol_store.py
"""CRUD operations for krag_symbol_index table."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}symbol_index"
FTS = f"{TABLE_PREFIX}symbol_fts"


def _build_fts_query(query: str) -> str | None:
    """OR-join alphanumeric words into FTS5 query syntax."""
    words = [w for w in re.findall(r"\w+", query) if w]
    if not words:
        return None
    return " OR ".join(words)


class SymbolStore:
    """CRUD for the symbol index."""

    def __init__(self, connection_manager: "SqliteConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, symbols: list[dict[str, Any]]) -> None:
        if not symbols:
            return

        sql = f"""
            INSERT INTO {TABLE}
                (id, name, qualified_name, kind, raw_file_id,
                 start_line, end_line, signature, summary,
                 imports, "references", keywords, entities, metadata)
            VALUES
                (?, ?, ?, ?, ?,
                 ?, ?, ?, ?,
                 ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                qualified_name = excluded.qualified_name,
                kind = excluded.kind,
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                signature = excluded.signature,
                summary = excluded.summary,
                imports = excluded.imports,
                "references" = excluded."references",
                keywords = excluded.keywords,
                entities = excluded.entities,
                metadata = excluded.metadata
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
                        json.dumps(sym.get("imports", [])),
                        json.dumps(sym.get("references", [])),
                        json.dumps(sym.get("keywords", [])),
                        json.dumps(sym.get("entities", [])),
                        json.dumps(sym.get("metadata", {})),
                    ),
                )
            conn.commit()

    def search_bm25(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """FTS5 BM25 search over symbol name + qualified_name + summary."""
        fts_query = _build_fts_query(query)
        if fts_query is None:
            return []

        sql = f"""
            SELECT s.id, s.name, s.qualified_name, s.kind, s.raw_file_id,
                   s.start_line, s.end_line, s.signature, s.summary, s.metadata,
                   bm25({FTS}) AS rank
            FROM {FTS}
            JOIN {TABLE} s ON s.rowid = {FTS}.rowid
            WHERE {FTS} MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (fts_query, limit)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:10])
            d["bm25_score"] = -float(row[10]) if row[10] is not None else 0.0
            results.append(d)
        return results

    def search_by_name(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Substring search against symbol name and qualified_name (case-insensitive)."""
        pattern = f"%{query}%"
        sql = f"""
            SELECT id, name, qualified_name, kind, raw_file_id,
                   start_line, end_line, signature, summary, metadata
            FROM {TABLE}
            WHERE name LIKE ? COLLATE NOCASE
               OR qualified_name LIKE ? COLLATE NOCASE
            LIMIT ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (pattern, pattern, limit)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def delete_by_file(self, raw_file_id: str) -> None:
        sql = f"DELETE FROM {TABLE} WHERE raw_file_id = ?"
        with self._cm.connection(self._collection) as conn:
            conn.execute(sql, (raw_file_id,))
            conn.commit()

    def get(self, symbol_id: str) -> dict[str, Any] | None:
        sql = f"""
            SELECT id, name, qualified_name, kind, raw_file_id,
                   start_line, end_line, signature, summary, metadata
            FROM {TABLE} WHERE id = ?
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (symbol_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        """All symbols for a file. ``references`` returned as a Python list."""
        sql = f"""
            SELECT id, name, qualified_name, kind, raw_file_id,
                   start_line, end_line, signature, summary, metadata,
                   "references"
            FROM {TABLE}
            WHERE raw_file_id = ?
            ORDER BY start_line
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:10])
            d["references"] = _decode_json_list(row[10])
            results.append(d)
        return results

    def search_by_keywords(self, terms: list[str], limit: int = 20) -> list[dict[str, Any]]:
        if not terms:
            return []
        placeholders = ",".join(["?"] * len(terms))
        sql = f"""
            SELECT id, name, qualified_name, kind, raw_file_id,
                   start_line, end_line, signature, summary, metadata
            FROM {TABLE}
            WHERE EXISTS (
                SELECT 1 FROM json_each({TABLE}.keywords) k
                WHERE k.value IN ({placeholders})
            )
            LIMIT ?
        """
        params = (*terms, limit)
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_structural_manifest(self) -> list[dict]:
        """Compact structural manifest of all symbols grouped by file."""
        sql = f"""
            SELECT s.raw_file_id, r.path,
                   s.name, s.qualified_name, s.kind, s.signature,
                   s.start_line, s.end_line, s.imports
            FROM {TABLE} s
            JOIN {TABLE_PREFIX}raw_files r ON s.raw_file_id = r.id
            ORDER BY r.path, s.start_line
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
                    "imports": _decode_json_list(row[8]),
                }
            )
        return list(files.values())

    def get_summaries_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        sql = f"""
            SELECT id, name, kind, summary
            FROM {TABLE}
            WHERE raw_file_id = ?
            ORDER BY start_line
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        return [{"id": row[0], "name": row[1], "kind": row[2], "summary": row[3]} for row in rows]

    def update_summaries_by_file(self, raw_file_id: str, summaries: list[str]) -> None:
        ids_sql = f"""
            SELECT id FROM {TABLE}
            WHERE raw_file_id = ?
            ORDER BY start_line
        """
        update_sql = f"UPDATE {TABLE} SET summary = ? WHERE id = ?"
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(ids_sql, (raw_file_id,)).fetchall()
            for i, row in enumerate(rows):
                if i < len(summaries):
                    conn.execute(update_sql, (summaries[i], row[0]))
            conn.commit()

    def update_enrichment_by_file(
        self, raw_file_id: str, enriched_dicts: list[dict[str, Any]]
    ) -> None:
        sql = f"UPDATE {TABLE} SET keywords = ?, entities = ?, metadata = ? WHERE id = ?"
        with self._cm.connection(self._collection) as conn:
            for item in enriched_dicts:
                conn.execute(
                    sql,
                    (
                        json.dumps(item.get("keywords", [])),
                        json.dumps(item.get("entities", [])),
                        json.dumps(item.get("metadata", {})),
                        item["id"],
                    ),
                )
            conn.commit()


def _decode_json_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return decoded if isinstance(decoded, list) else []


def _row_to_dict(row: tuple) -> dict[str, Any]:
    meta = row[9]
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
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
