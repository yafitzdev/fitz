# fitz_ai/engines/fitz_rag/routing/router.py
"""
Query Router - Routes queries to appropriate hierarchy level.

Global queries (trends, summaries) -> L2 corpus summaries
Local queries (specific facts) -> L0 chunks (default behavior)

Uses semantic embeddings for language-agnostic classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from fitz_ai.core.guardrails.semantic import cosine_similarity
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Type alias for embedder function
EmbedderFunc = Callable[[str], list[float]]


class QueryIntent(Enum):
    """Query intent classification."""

    GLOBAL = "global"  # Trends, summaries, overviews -> L2
    LOCAL = "local"  # Specific facts, details -> L0


# Exemplar queries that represent "global" intent
# The embedder maps semantically similar queries to nearby vectors
GLOBAL_QUERY_EXEMPLARS: tuple[str, ...] = (
    "give me an overview of this",
    "summarize the main topics",
    "what are the key themes",
    "what is the big picture",
    "what are the trends",
    "tell me about this project",
    "what is this about",
    "what are the main points",
    "provide a high-level summary",
    "what patterns do you see across all",
    "give me a general sense",
    "what are the overall findings",
)


@dataclass
class QueryRouter:
    """
    Routes queries to appropriate retrieval targets based on intent.

    Uses semantic embeddings to classify queries as global or local.
    Global queries are routed to L2 corpus summaries.
    Local queries use standard L0 chunk retrieval.

    Attributes:
        enabled: Whether routing is active (default True)
        embedder: Embedding function for semantic classification
        threshold: Similarity threshold for GLOBAL classification (default 0.7)
    """

    enabled: bool = True
    embedder: EmbedderFunc | None = None
    threshold: float = 0.7

    # Internal cache for exemplar vectors
    _exemplar_cache: list[list[float]] | None = field(
        default=None, repr=False, compare=False
    )

    def _get_exemplar_vectors(self) -> list[list[float]]:
        """Get cached exemplar vectors or compute and cache them."""
        if self._exemplar_cache is None and self.embedder is not None:
            self._exemplar_cache = [self.embedder(e) for e in GLOBAL_QUERY_EXEMPLARS]
        return self._exemplar_cache or []

    def _max_similarity(self, query_vec: list[float]) -> float:
        """Find maximum similarity between query and any global exemplar."""
        exemplar_vecs = self._get_exemplar_vectors()
        if not exemplar_vecs:
            return 0.0
        return max(cosine_similarity(query_vec, ev) for ev in exemplar_vecs)

    def classify(self, query: str) -> QueryIntent:
        """
        Classify query as global or local using semantic similarity.

        Args:
            query: The query text to classify

        Returns:
            QueryIntent.GLOBAL for analytical queries (trends, summaries)
            QueryIntent.LOCAL for specific fact queries
        """
        if not self.enabled:
            return QueryIntent.LOCAL

        if self.embedder is None:
            logger.debug("QueryRouter: no embedder provided, defaulting to LOCAL")
            return QueryIntent.LOCAL

        # Embed query and compare to global exemplars
        query_vec = self.embedder(query)
        similarity = self._max_similarity(query_vec)

        if similarity >= self.threshold:
            logger.debug(
                f"QueryRouter: '{query[:50]}...' -> GLOBAL (similarity={similarity:.3f})"
            )
            return QueryIntent.GLOBAL

        logger.debug(
            f"QueryRouter: '{query[:50]}...' -> LOCAL (similarity={similarity:.3f})"
        )
        return QueryIntent.LOCAL

    def get_l2_filter(self) -> dict[str, Any]:
        """
        Return filter for L2 corpus summaries.

        Returns:
            Qdrant-style filter dict targeting hierarchy summaries
        """
        return {
            "must": [
                {"key": "is_hierarchy_summary", "match": {"value": True}},
            ]
        }
