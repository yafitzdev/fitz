# core/vector_db/plugins/qdrant.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from fitz.core.vector_db.base import VectorDBPlugin


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    plugin_name: str = "qdrant"
    plugin_type: str = "vector_db"

    host: str = "localhost"
    port: int = 6333

    def __post_init__(self) -> None:
        self._client = QdrantClient(host=self.host, port=self.port)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[Any]:
        return self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload,
        )

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        q_points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload"),
            )
            for p in points
        ]
        self._client.upsert(collection_name=collection, points=q_points)
