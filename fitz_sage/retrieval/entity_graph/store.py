# fitz_sage/retrieval/entity_graph/store.py
"""Persistent entity-to-chunk graph for related chunk discovery (SQLite)."""

from __future__ import annotations

from fitz_sage.logging.logger import get_logger
from fitz_sage.logging.tags import STORAGE
from fitz_sage.storage import get_connection_manager

logger = get_logger(__name__)


class EntityGraphStore:
    """Persistent entity-to-chunk graph using SQLite.

    Per-collection storage alongside the krag tables in the collection's
    ``.db`` file.
    """

    SCHEMA_SQL = """
        CREATE TABLE IF NOT EXISTS entities (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            entity_type TEXT,
            mention_count INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS entity_chunks (
            entity_name TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            PRIMARY KEY (entity_name, chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_chunk_entities
        ON entity_chunks(chunk_id);

        CREATE INDEX IF NOT EXISTS idx_entity_chunks
        ON entity_chunks(entity_name);
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
            conn.executescript(self.SCHEMA_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug(f"{STORAGE} Entity graph schema initialized for '{self.collection}'")

    # =========================================================================
    # Write Operations
    # =========================================================================

    def add_chunk_entities(
        self,
        chunk_id: str,
        entities: list[tuple[str, str]],
    ) -> None:
        if not entities:
            return

        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            for display_name, entity_type in entities:
                normalized = self._normalize(display_name)
                if not normalized:
                    continue

                conn.execute(
                    """
                    INSERT INTO entities (name, display_name, entity_type, mention_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(name) DO UPDATE SET mention_count = entities.mention_count + 1
                    """,
                    (normalized, display_name, entity_type),
                )

                conn.execute(
                    """
                    INSERT INTO entity_chunks (entity_name, chunk_id)
                    VALUES (?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (normalized, chunk_id),
                )

            conn.commit()

        logger.debug(f"Added {len(entities)} entities for chunk {chunk_id[:8]}...")

    def remove_chunk(self, chunk_id: str) -> None:
        self._ensure_schema()
        with self._manager.connection(self.collection) as conn:
            conn.execute("DELETE FROM entity_chunks WHERE chunk_id = ?", (chunk_id,))
            conn.commit()

    # =========================================================================
    # Read Operations
    # =========================================================================

    def get_related_chunks(
        self,
        chunk_ids: list[str],
        max_total: int = 20,
        min_shared_entities: int = 1,
    ) -> list[str]:
        if not chunk_ids:
            return []

        self._ensure_schema()

        chunk_ph = ",".join(["?"] * len(chunk_ids))
        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute(
                f"""
                SELECT DISTINCT entity_name
                FROM entity_chunks
                WHERE chunk_id IN ({chunk_ph})
                """,
                tuple(chunk_ids),
            )
            entities = [row[0] for row in cursor.fetchall()]

            if not entities:
                return []

            entity_ph = ",".join(["?"] * len(entities))
            cursor = conn.execute(
                f"""
                SELECT chunk_id, COUNT(DISTINCT entity_name) AS shared_count
                FROM entity_chunks
                WHERE entity_name IN ({entity_ph})
                  AND chunk_id NOT IN ({chunk_ph})
                GROUP BY chunk_id
                HAVING shared_count >= ?
                ORDER BY shared_count DESC
                LIMIT ?
                """,
                (*entities, *chunk_ids, min_shared_entities, max_total),
            )

            return [row[0] for row in cursor.fetchall()]

    def get_chunks_for_entity(self, entity: str, limit: int = 10) -> list[str]:
        self._ensure_schema()
        normalized = self._normalize(entity)

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute(
                """
                SELECT chunk_id FROM entity_chunks
                WHERE entity_name = ?
                LIMIT ?
                """,
                (normalized, limit),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_chunks_for_entities(
        self,
        entities: list[str],
        limit: int = 20,
    ) -> list[str]:
        if not entities:
            return []

        self._ensure_schema()
        normalized = [self._normalize(e) for e in entities if self._normalize(e)]
        if not normalized:
            return []

        placeholders = ",".join(["?"] * len(normalized))
        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute(
                f"""
                SELECT chunk_id, COUNT(DISTINCT entity_name) AS match_count
                FROM entity_chunks
                WHERE entity_name IN ({placeholders})
                GROUP BY chunk_id
                ORDER BY match_count DESC
                LIMIT ?
                """,
                (*normalized, limit),
            )
            return [row[0] for row in cursor.fetchall()]

    def find_related_topics(
        self,
        terms: list[str],
        limit: int = 5,
    ) -> list[dict]:
        if not terms:
            return []

        self._ensure_schema()

        normalized = [self._normalize(t) for t in terms if self._normalize(t)]
        if not normalized:
            return []

        like_conditions = " OR ".join(["name LIKE ?"] * len(normalized))
        like_params = [f"%{t}%" for t in normalized]

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute(
                f"""
                SELECT display_name, entity_type, mention_count
                FROM entities
                WHERE {like_conditions}
                ORDER BY mention_count DESC
                LIMIT ?
                """,
                (*like_params, limit),
            )
            return [
                {"name": row[0], "type": row[1], "mentions": row[2]} for row in cursor.fetchall()
            ]

    # =========================================================================
    # Utilities
    # =========================================================================

    def _normalize(self, name: str) -> str:
        if not name:
            return ""
        return name.lower().strip()

    def stats(self) -> dict:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM entity_chunks").fetchone()[0]

            top_entities = conn.execute(
                """
                SELECT display_name, mention_count
                FROM entities
                ORDER BY mention_count DESC
                LIMIT 10
                """
            ).fetchall()

            return {
                "entities": entities,
                "edges": edges,
                "top_entities": [{"name": row[0], "mentions": row[1]} for row in top_entities],
            }

    def clear(self) -> None:
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            conn.execute("DELETE FROM entity_chunks")
            conn.execute("DELETE FROM entities")
            conn.commit()

        logger.info(f"Cleared entity graph for collection {self.collection}")

    def close(self) -> None:
        """No-op (no persistent connection)."""
        pass
