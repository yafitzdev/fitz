# fitz_ai/map/state.py
"""
State manager for knowledge map.

Handles loading, saving, and updating the .fitz/knowledge_map.json state file.

Key responsibilities:
- Load/create state file
- Cache chunk embeddings (float16)
- Invalidate cache when collection or embedding config changes
- Remove stale chunks when they're deleted from vector DB
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fitz_ai.core.paths import FitzPaths
from fitz_ai.map.models import ChunkEmbedding, KnowledgeMapState

logger = logging.getLogger(__name__)


class KnowledgeMapStateManager:
    """
    Manages the knowledge map state file.

    The state file caches:
    - Chunk embeddings (float16) for incremental updates
    - Document hierarchy for visualization
    - Collection and embedding config for cache invalidation

    Usage:
        manager = KnowledgeMapStateManager()
        state = manager.load(collection="default", embedding_id="ollama:nomic-embed-text")

        # Add new embeddings
        manager.add_chunk(chunk_embedding)

        # Remove stale chunks
        removed = manager.remove_stale_chunks(current_chunk_ids)

        manager.save()
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        """
        Initialize the state manager.

        Args:
            path: Path to state file. Defaults to .fitz/knowledge_map.json
        """
        self._path = path or FitzPaths.knowledge_map()
        self._state: Optional[KnowledgeMapState] = None
        self._dirty: bool = False

    @property
    def state(self) -> KnowledgeMapState:
        """Get the current state. Raises if not loaded."""
        if self._state is None:
            raise RuntimeError("State not loaded. Call load() first.")
        return self._state

    def load(self, collection: str, embedding_id: str) -> KnowledgeMapState:
        """
        Load state from disk, or create new if not exists.

        Invalidates cache if collection or embedding_id changed.

        Args:
            collection: Vector DB collection name.
            embedding_id: Embedding config ID (e.g., "ollama:nomic-embed-text").

        Returns:
            The loaded or created state.
        """
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._state = KnowledgeMapState.model_validate(data)
                logger.debug(f"Loaded knowledge map state from {self._path}")

                # Check if cache should be invalidated
                if self._should_invalidate_cache(collection, embedding_id):
                    logger.info(
                        f"Cache invalidated: collection or embedding changed. "
                        f"Old: {self._state.collection}/{self._state.embedding_id}, "
                        f"New: {collection}/{embedding_id}"
                    )
                    self._state = self._create_new_state(collection, embedding_id)
                    self._dirty = True

            except Exception as e:
                logger.warning(f"Failed to load state, creating new: {e}")
                self._state = self._create_new_state(collection, embedding_id)
                self._dirty = True
        else:
            self._state = self._create_new_state(collection, embedding_id)
            self._dirty = True
            logger.debug("Created new knowledge map state")

        return self._state

    def _should_invalidate_cache(self, collection: str, embedding_id: str) -> bool:
        """Check if cache should be invalidated due to config changes."""
        if self._state is None:
            return True
        return (
            self._state.collection != collection
            or self._state.embedding_id != embedding_id
        )

    def save(self) -> None:
        """
        Save state to disk.

        Only writes if there are unsaved changes.
        """
        if self._state is None:
            return

        if not self._dirty:
            logger.debug("State unchanged, skipping save")
            return

        # Update timestamp
        self._state.updated_at = datetime.utcnow()

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically using temp file
        temp_path = self._path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(
                    self._state.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,
                )
            temp_path.replace(self._path)
            self._dirty = False
            logger.debug(f"Saved knowledge map state to {self._path}")
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _create_new_state(
        self, collection: str, embedding_id: str
    ) -> KnowledgeMapState:
        """Create a new empty state."""
        return KnowledgeMapState(
            collection=collection,
            embedding_id=embedding_id,
            updated_at=datetime.utcnow(),
        )

    def get_cached_chunk_ids(self) -> set[str]:
        """Get set of chunk IDs already in cache."""
        return self.state.get_cached_chunk_ids()

    def add_chunk(self, chunk: ChunkEmbedding) -> None:
        """Add a chunk embedding to the cache."""
        self.state.add_chunk(chunk)
        self._dirty = True

    def add_chunks(self, chunks: list[ChunkEmbedding]) -> None:
        """Add multiple chunk embeddings to the cache."""
        for chunk in chunks:
            self.state.add_chunk(chunk)
        if chunks:
            self._dirty = True

    def remove_stale_chunks(self, current_chunk_ids: set[str]) -> int:
        """
        Remove chunks that no longer exist in the vector DB.

        Args:
            current_chunk_ids: Set of chunk IDs currently in vector DB.

        Returns:
            Number of chunks removed.
        """
        removed = self.state.remove_stale_chunks(current_chunk_ids)
        if removed > 0:
            self._dirty = True
            logger.info(f"Removed {removed} stale chunks from cache")
        return removed

    def get_chunks_needing_fetch(self, current_chunk_ids: set[str]) -> set[str]:
        """
        Get chunk IDs that need to be fetched from vector DB.

        Args:
            current_chunk_ids: Set of chunk IDs currently in vector DB.

        Returns:
            Set of chunk IDs not in cache.
        """
        cached = self.get_cached_chunk_ids()
        return current_chunk_ids - cached

    def mark_dirty(self) -> None:
        """Mark state as needing save."""
        self._dirty = True


__all__ = ["KnowledgeMapStateManager"]
