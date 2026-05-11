# fitz_sage/retrieval/vocabulary/store.py
"""
Vocabulary store for persisting keywords to SQLite.

Per-collection vocabulary stored alongside the krag tables in the
collection's ``.db`` file. User modifications are preserved across
re-ingests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fitz_sage.logging.logger import get_logger
from fitz_sage.logging.tags import STORAGE
from fitz_sage.storage import get_connection_manager

from .models import Keyword, VocabularyMetadata

logger = get_logger(__name__)


class VocabularyStore:
    """Manages keyword vocabulary persistence in SQLite."""

    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS keywords (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            match TEXT NOT NULL DEFAULT '[]',
            occurrences INTEGER NOT NULL DEFAULT 1,
            first_seen TEXT,
            user_defined INTEGER NOT NULL DEFAULT 0,
            auto_generated TEXT NOT NULL DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS vocabulary_meta (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            generated TEXT NOT NULL,
            source_docs INTEGER NOT NULL DEFAULT 0,
            auto_detected INTEGER NOT NULL DEFAULT 0,
            user_modified INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_keywords_category
        ON keywords(category);
    """

    def __init__(self, collection: str | None = None, path=None):
        self.collection = collection or "default"
        self._manager = get_connection_manager()
        self._manager.start()
        self._schema_initialized = False

    def _ensure_schema(self) -> None:
        if self._schema_initialized:
            return

        with self._manager.connection(self.collection) as conn:
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug(f"{STORAGE} Vocabulary schema initialized for '{self.collection}'")

    def exists(self) -> bool:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            result = conn.execute("SELECT COUNT(*) FROM keywords").fetchone()
            return result[0] > 0 if result else False

    def load(self) -> list[Keyword]:
        self._ensure_schema()
        try:
            with self._manager.connection(self.collection) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, category, match, occurrences, first_seen,
                           user_defined, auto_generated
                    FROM keywords
                    ORDER BY category, id
                    """
                )

                keywords = []
                for row in cursor.fetchall():
                    keywords.append(
                        Keyword(
                            id=row[0],
                            category=row[1],
                            match=_decode_list(row[2]),
                            occurrences=row[3],
                            first_seen=row[4],
                            user_defined=bool(row[5]),
                            auto_generated=_decode_list(row[6]),
                        )
                    )

                logger.debug(
                    f"[VOCABULARY] Loaded {len(keywords)} keywords for '{self.collection}'"
                )
                return keywords

        except Exception as e:
            logger.warning(f"[VOCABULARY] Failed to load keywords: {e}")
            return []

    def load_with_metadata(self) -> tuple[list[Keyword], VocabularyMetadata | None]:
        self._ensure_schema()
        try:
            keywords = self.load()

            with self._manager.connection(self.collection) as conn:
                result = conn.execute(
                    """
                    SELECT generated, source_docs, auto_detected, user_modified
                    FROM vocabulary_meta WHERE id = 1
                    """
                ).fetchone()

                if result:
                    generated = result[0]
                    if isinstance(generated, str):
                        try:
                            generated = datetime.fromisoformat(generated)
                        except ValueError:
                            generated = datetime.now(timezone.utc)
                    metadata = VocabularyMetadata(
                        generated=generated or datetime.now(timezone.utc),
                        source_docs=result[1],
                        auto_detected=result[2],
                        user_modified=result[3],
                    )
                else:
                    metadata = None

            return keywords, metadata

        except Exception as e:
            logger.warning(f"[VOCABULARY] Failed to load keywords: {e}")
            return [], None

    def save(
        self,
        keywords: list[Keyword],
        metadata: VocabularyMetadata | None = None,
    ) -> None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            conn.execute("DELETE FROM keywords")

            for kw in keywords:
                conn.execute(
                    """
                    INSERT INTO keywords (id, category, match, occurrences, first_seen,
                                         user_defined, auto_generated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        kw.id,
                        kw.category,
                        json.dumps(list(kw.match)),
                        kw.occurrences,
                        kw.first_seen,
                        1 if kw.user_defined else 0,
                        json.dumps(list(kw.auto_generated)),
                    ),
                )

            if not metadata:
                metadata = VocabularyMetadata(
                    auto_detected=len([k for k in keywords if not k.user_defined]),
                    user_modified=len([k for k in keywords if k.user_defined]),
                )

            generated = metadata.generated
            generated_str = (
                generated.isoformat()
                if hasattr(generated, "isoformat")
                else str(generated) if generated else datetime.now(timezone.utc).isoformat()
            )

            conn.execute(
                """
                INSERT INTO vocabulary_meta (id, generated, source_docs, auto_detected, user_modified)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    generated = excluded.generated,
                    source_docs = excluded.source_docs,
                    auto_detected = excluded.auto_detected,
                    user_modified = excluded.user_modified
                """,
                (
                    generated_str,
                    metadata.source_docs,
                    metadata.auto_detected,
                    metadata.user_modified,
                ),
            )

            conn.commit()

        logger.info(f"[VOCABULARY] Saved {len(keywords)} keywords for '{self.collection}'")

    def merge_and_save(
        self,
        new_keywords: list[Keyword],
        source_docs: int = 0,
    ) -> list[Keyword]:
        existing, _ = self.load_with_metadata()
        existing_by_id: dict[str, Keyword] = {kw.id.lower(): kw for kw in existing}
        merged: list[Keyword] = []

        for new_kw in new_keywords:
            key = new_kw.id.lower()

            if key in existing_by_id:
                old_kw = existing_by_id[key]
                user_variations = set(old_kw.match) - set(old_kw.auto_generated)
                all_variations = set(new_kw.match) | user_variations
                new_kw.match = sorted(all_variations, key=str.lower)
                new_kw.auto_generated = new_kw.match.copy()
                new_kw.user_defined = old_kw.user_defined
                new_kw.occurrences = max(new_kw.occurrences, old_kw.occurrences)
                del existing_by_id[key]

            merged.append(new_kw)

        for old_kw in existing_by_id.values():
            if old_kw.user_defined:
                merged.append(old_kw)

        auto_detected = len([k for k in merged if not k.user_defined])
        user_modified = len([k for k in merged if k.user_defined])

        metadata = VocabularyMetadata(
            generated=datetime.now(timezone.utc),
            source_docs=source_docs,
            auto_detected=auto_detected,
            user_modified=user_modified,
        )

        self.save(merged, metadata)
        return merged

    def add_keyword(self, keyword: Keyword) -> None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT id FROM keywords WHERE LOWER(id) = LOWER(?)",
                (keyword.id,),
            ).fetchone()

            if result:
                logger.warning(f"[VOCABULARY] Keyword {keyword.id!r} already exists")
                return

            keyword.user_defined = True
            conn.execute(
                """
                INSERT INTO keywords (id, category, match, occurrences, first_seen,
                                     user_defined, auto_generated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    keyword.id,
                    keyword.category,
                    json.dumps(list(keyword.match)),
                    keyword.occurrences,
                    keyword.first_seen,
                    1 if keyword.user_defined else 0,
                    json.dumps(list(keyword.auto_generated)),
                ),
            )
            conn.commit()

        logger.info(f"[VOCABULARY] Added keyword {keyword.id!r}")

    def add_variation(self, keyword_id: str, variation: str) -> bool:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT match FROM keywords WHERE LOWER(id) = LOWER(?)",
                (keyword_id,),
            ).fetchone()

            if not result:
                logger.warning(f"[VOCABULARY] Keyword {keyword_id!r} not found")
                return False

            current_match = _decode_list(result[0])

            if variation not in current_match:
                current_match.append(variation)
                conn.execute(
                    "UPDATE keywords SET match = ? WHERE LOWER(id) = LOWER(?)",
                    (json.dumps(current_match), keyword_id),
                )
                conn.commit()
                logger.info(f"[VOCABULARY] Added variation {variation!r} to {keyword_id}")

            return True

    def remove_keyword(self, keyword_id: str) -> bool:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            existed = conn.execute(
                "SELECT 1 FROM keywords WHERE LOWER(id) = LOWER(?)",
                (keyword_id,),
            ).fetchone()
            if not existed:
                logger.warning(f"[VOCABULARY] Keyword {keyword_id!r} not found")
                return False
            conn.execute(
                "DELETE FROM keywords WHERE LOWER(id) = LOWER(?)",
                (keyword_id,),
            )
            conn.commit()
            logger.info(f"[VOCABULARY] Removed keyword {keyword_id!r}")
            return True

    def get_by_category(self, category: str) -> list[Keyword]:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute(
                """
                SELECT id, category, match, occurrences, first_seen,
                       user_defined, auto_generated
                FROM keywords
                WHERE category = ?
                ORDER BY id
                """,
                (category,),
            )

            return [
                Keyword(
                    id=row[0],
                    category=row[1],
                    match=_decode_list(row[2]),
                    occurrences=row[3],
                    first_seen=row[4],
                    user_defined=bool(row[5]),
                    auto_generated=_decode_list(row[6]),
                )
                for row in cursor.fetchall()
            ]

    def get_categories(self) -> list[str]:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute("SELECT DISTINCT category FROM keywords ORDER BY category")
            return [row[0] for row in cursor.fetchall()]

    def clear(self) -> None:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            conn.execute("DELETE FROM keywords")
            conn.execute("DELETE FROM vocabulary_meta")
            conn.commit()
        logger.info(f"[VOCABULARY] Cleared vocabulary for '{self.collection}'")


def _decode_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return decoded if isinstance(decoded, list) else []
