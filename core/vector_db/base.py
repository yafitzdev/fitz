from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class VectorRecord:
    """
    A single vector record to be upserted into a vector DB.
    """
    id: Any
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """
    Normalized search result from a vector DB.
    """
    id: Any
    score: float
    payload: Dict[str, Any]


class VectorDBPlugin(Protocol):
    """
    Minimal protocol for vector DB plugins.

    Implementations should be lightweight wrappers around a concrete
    vector DB client (e.g., Qdrant, Chroma, etc.).
    """

    # Plugin registry name (e.g., "qdrant")
    plugin_name: str

    def upsert(self, collection: str, records: List[VectorRecord]) -> None:
        """
        Insert or update a batch of vector records into the given collection.
        """
        ...

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        with_payload: bool = True,
    ) -> List[Any]:
        """
        Run a vector search.

        IMPORTANT:
        ---------
        The return type is intentionally `List[Any]` here so that
        retrievers can work directly with backend-specific objects
        (e.g., Qdrant `ScoredPoint`), as long as they provide:

            - `.id`
            - `.score`
            - `.payload` (or a dict-like with "text" etc.)

        This keeps `DenseRetrievalPlugin` compatible with both
        real clients and the dummy clients used in tests.
        """
        ...

    def delete_collection(self, collection: str) -> None:
        """
        Drop the given collection (if supported by the backend).
        """
        ...
