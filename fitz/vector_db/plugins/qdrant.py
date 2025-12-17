# fitz/vector_db/plugins/qdrant.py
"""
Qdrant vector database plugin.

Supports qdrant-client >= 1.7 API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from fitz.vector_db.base import VectorDBPlugin, SearchResult


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    """
    Qdrant vector database plugin.

    Config example:
        vector_db:
          plugin_name: qdrant
          kwargs:
            host: localhost
            port: 6333
    """

    plugin_name: str = "qdrant"
    plugin_type: str = "vector_db"

    host: str = "localhost"
    port: int = 6333

    _client: QdrantClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = QdrantClient(host=self.host, port=self.port)

    def search(
            self,
            collection_name: str,
            query_vector: list[float],
            limit: int,
            with_payload: bool = True,
    ) -> list[SearchResult]:
        """
        Search for similar vectors in a collection.

        Uses qdrant-client >= 1.7 API (query_points).
        """
        # qdrant-client >= 1.7 uses query_points
        result = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,  # This is the vector, not text!
            limit=limit,
            with_payload=with_payload,
        )

        # Convert to SearchResult objects
        return [
            SearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in result.points
        ]

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """Upsert points into a collection."""
        q_points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload"),
            )
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=q_points)