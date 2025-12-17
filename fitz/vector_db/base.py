# core/vector_db/base.py
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
    plugin_name: str
    plugin_type: str  # must be "vector_db"

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[SearchResult] | list[Any]: ...
