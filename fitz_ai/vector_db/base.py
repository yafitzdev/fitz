# fitz_ai/vector_db/base.py
"""
Base types for vector database plugins.

This module defines the canonical types that all vector DB implementations
must conform to, regardless of whether they're YAML-based or Python-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Canonical vector search hit shape.

    Backends may return richer objects, but RAG retrieval expects at minimum:
    - id
    - score
    - payload (dict)
    """

    id: str
    score: float | None
    payload: dict[str, Any]


@runtime_checkable
class VectorDBPlugin(Protocol):
    """
    Protocol for vector database plugins.

    All vector DB implementations (YAML-based, Python-based, custom) must
    implement this interface to be usable by the RAG pipeline.
    """

    plugin_name: str
    plugin_type: str  # must be "vector_db"

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[SearchResult]:
        """
        Search for similar vectors in collection.

        Args:
            collection_name: Name of the collection to search
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            with_payload: Whether to include payload in results

        Returns:
            List of SearchResult objects, ordered by similarity (highest first)
        """
        ...
