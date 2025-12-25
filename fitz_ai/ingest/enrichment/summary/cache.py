# fitz_ai/ingest/enrichment/summary/cache.py
"""
Caching for chunk summaries.

The cache stores LLM-generated descriptions keyed by:
- content_hash: Hash of the chunk content
- enricher_id: Identifier of the enricher configuration

This ensures descriptions are regenerated when:
1. The content changes (different hash)
2. The enricher changes (different model, prompt version, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SummaryCache:
    """
    Persistent cache for chunk summaries.

    Caches LLM-generated descriptions to avoid redundant API calls.
    Cache keys combine content hash and enricher ID to ensure
    descriptions are regenerated when either changes.

    Usage:
        cache = SummaryCache(Path(".fitz/enrichment_cache.json"))

        desc = cache.get(content_hash="abc123", enricher_id="llm:gpt-4o:v1")
        if desc is None:
            desc = summarizer.summarize(content, context)
            cache.set(content_hash="abc123", enricher_id="llm:gpt-4o:v1", description=desc)
    """

    def __init__(self, cache_path: Path | str):
        self._path = Path(cache_path)
        self._cache: Dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached summaries")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load summary cache: {e}")
                self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        if not self._dirty:
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            self._dirty = False
            logger.debug(f"Saved {len(self._cache)} cached summaries")
        except IOError as e:
            logger.warning(f"Failed to save summary cache: {e}")

    @staticmethod
    def _make_key(content_hash: str, enricher_id: str) -> str:
        """Create cache key from content hash and enricher ID."""
        return f"{content_hash}:{enricher_id}"

    def get(self, content_hash: str, enricher_id: str) -> Optional[str]:
        """Get cached description if available."""
        key = self._make_key(content_hash, enricher_id)
        return self._cache.get(key)

    def set(self, content_hash: str, enricher_id: str, description: str) -> None:
        """Store a description in the cache."""
        key = self._make_key(content_hash, enricher_id)
        self._cache[key] = description
        self._dirty = True

    def save(self) -> None:
        """Explicitly save cache to disk."""
        self._save()

    def clear(self) -> None:
        """Clear all cached descriptions."""
        self._cache = {}
        self._dirty = True
        self._save()

    def remove(self, content_hash: str, enricher_id: str) -> bool:
        """Remove a specific entry from the cache."""
        key = self._make_key(content_hash, enricher_id)
        if key in self._cache:
            del self._cache[key]
            self._dirty = True
            return True
        return False

    def __len__(self) -> int:
        """Return number of cached descriptions."""
        return len(self._cache)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check if a (content_hash, enricher_id) pair is cached."""
        content_hash, enricher_id = key
        return self._make_key(content_hash, enricher_id) in self._cache

    def __enter__(self) -> "SummaryCache":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - save on exit."""
        self._save()


__all__ = ["SummaryCache"]
