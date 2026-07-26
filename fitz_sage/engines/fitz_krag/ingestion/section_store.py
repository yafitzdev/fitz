# fitz_sage/engines/fitz_krag/ingestion/section_store.py
"""CRUD operations for krag_section_index table with FTS5 BM25 search."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion import store_utils
from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    from fitz_sage.storage.sqlite import SqliteConnectionManager

logger = logging.getLogger(__name__)

TABLE = f"{TABLE_PREFIX}section_index"
FTS = f"{TABLE_PREFIX}section_fts"
CORPUS_SUMMARY_SCHEMA_VERSION = 2


class SectionStore:
    """CRUD for the section index."""

    def __init__(self, connection_manager: "SqliteConnectionManager", collection: str):
        self._cm = connection_manager
        self._collection = collection

    def upsert_batch(self, sections: list[dict[str, Any]]) -> None:
        if not sections:
            return

        sql = f"""
            INSERT INTO {TABLE}
                (id, raw_file_id, title, level, page_start, page_end,
                 content, summary, parent_section_id,
                 position, keywords, entities, metadata)
            VALUES
                (?, ?, ?, ?, ?, ?,
                 ?, ?, ?,
                 ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                level = excluded.level,
                page_start = excluded.page_start,
                page_end = excluded.page_end,
                content = excluded.content,
                summary = excluded.summary,
                parent_section_id = excluded.parent_section_id,
                position = excluded.position,
                keywords = excluded.keywords,
                entities = excluded.entities,
                metadata = excluded.metadata
        """
        with self._cm.connection(self._collection) as conn:
            for sec in sections:
                conn.execute(
                    sql,
                    (
                        sec["id"],
                        sec["raw_file_id"],
                        sec["title"],
                        sec["level"],
                        sec.get("page_start"),
                        sec.get("page_end"),
                        sec["content"],
                        sec.get("summary"),
                        sec.get("parent_section_id"),
                        sec["position"],
                        json.dumps(sec.get("keywords", [])),
                        json.dumps(sec.get("entities", [])),
                        json.dumps(sec.get("metadata", {})),
                    ),
                )
            conn.commit()

    def search_bm25(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search using FTS5 + bm25 ranking.

        This does a direct section match; downstream ``_score_results``
        applies RRF on the returned order, and parent-title breadcrumbs
        are pulled in by ``SectionSearchStrategy._enrich_with_parent_titles``.
        """
        fts_query = store_utils.build_fts_query(query)
        if fts_query is None:
            return []

        sql = f"""
            SELECT s.id, s.raw_file_id, s.title, s.level,
                   s.page_start, s.page_end, s.content, s.summary,
                   s.parent_section_id, s.position, s.keywords, s.entities, s.metadata,
                   bm25({FTS}) AS rank
            FROM {FTS}
            JOIN {TABLE} s ON s.rowid = {FTS}.rowid
            WHERE {FTS} MATCH ?
              AND (
                  json_extract(s.metadata, '$.is_corpus_summary') IS NULL
                  OR (
                      json_extract(s.metadata, '$.is_corpus_summary') != 'true'
                      AND json_extract(s.metadata, '$.is_corpus_summary') != 1
                  )
              )
            ORDER BY rank
            LIMIT ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (fts_query, limit)).fetchall()
        results = []
        for row in rows:
            d = _row_to_dict(row[:13])
            # bm25() returns negative numbers (lower=better); flip sign so
            # downstream code that treats higher-better is consistent.
            d["bm25_score"] = -float(row[13]) if row[13] is not None else 0.0
            results.append(d)
        return results

    def get(self, section_id: str) -> dict[str, Any] | None:
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, keywords, entities, metadata
            FROM {TABLE} WHERE id = ?
        """
        with self._cm.connection(self._collection) as conn:
            row = conn.execute(sql, (section_id,)).fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    def get_by_file(self, raw_file_id: str) -> list[dict[str, Any]]:
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, keywords, entities, metadata
            FROM {TABLE}
            WHERE raw_file_id = ?
            ORDER BY position
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (raw_file_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def search_by_keywords(self, terms: list[str], limit: int = 20) -> list[dict[str, Any]]:
        """Find sections with matching enriched keywords.

        Keywords are stored as JSON; we expand them via ``json_each`` and
        match against the term list with an IN clause.
        """
        if not terms:
            return []
        placeholders = ",".join(["?"] * len(terms))
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, keywords, entities, metadata
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

    def get_children(self, section_id: str) -> list[dict[str, Any]]:
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, keywords, entities, metadata
            FROM {TABLE}
            WHERE parent_section_id = ?
            ORDER BY position
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (section_id,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def update_summaries_by_file(self, raw_file_id: str, summaries: list[str]) -> None:
        ids_sql = f"""
            SELECT id FROM {TABLE}
            WHERE raw_file_id = ?
            ORDER BY position
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
        store_utils.update_enrichment_by_file(self._cm, self._collection, TABLE, enriched_dicts)

    def get_corpus_summaries(self) -> list[dict[str, Any]]:
        """Fetch all L2 corpus-level summary sections for this collection."""
        sql = f"""
            SELECT id, raw_file_id, title, level, page_start, page_end,
                   content, summary, parent_section_id, position, keywords, entities, metadata
            FROM {TABLE}
            WHERE (
                json_extract(metadata, '$.is_corpus_summary') = 'true'
                OR json_extract(metadata, '$.is_corpus_summary') = 1
            )
            AND json_extract(metadata, '$.corpus_summary_schema') = ?
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql, (CORPUS_SUMMARY_SCHEMA_VERSION,)).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_hierarchy_summaries(self) -> list[str]:
        """Distinct L1 hierarchy summaries across all sections.

        Each document file contributes one L1 group summary, repeated on the
        metadata of every section in that file. The corpus ``finalize`` step
        rolls these up into the L2 summary.
        """
        sql = f"""
            SELECT DISTINCT json_extract(metadata, '$.hierarchy_summary')
            FROM {TABLE}
            WHERE json_extract(metadata, '$.hierarchy_summary') IS NOT NULL
        """
        with self._cm.connection(self._collection) as conn:
            rows = conn.execute(sql).fetchall()
        return [row[0] for row in rows if row[0]]

    def delete_by_file(self, raw_file_id: str) -> None:
        store_utils.delete_by_file(self._cm, self._collection, TABLE, raw_file_id)


def _row_to_dict(row: tuple) -> dict[str, Any]:
    keywords = store_utils.decode_json(row[10], [])
    entities = store_utils.decode_json(row[11], [])
    meta = store_utils.decode_json(row[12], {})
    return {
        "id": row[0],
        "raw_file_id": row[1],
        "title": row[2],
        "level": row[3],
        "page_start": row[4],
        "page_end": row[5],
        "content": row[6],
        "summary": row[7],
        "parent_section_id": row[8],
        "position": row[9],
        "keywords": keywords if isinstance(keywords, list) else [],
        "entities": entities if isinstance(entities, list) else [],
        "metadata": meta,
    }
