from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.vector_db.base import VectorDBPlugin, VectorRecord


@dataclass
class VectorDBEngine:
    """
    Thin orchestration wrapper around a VectorDBPlugin.

    Keeps ingestion code plugin-agnostic and testable.
    """

    plugin: VectorDBPlugin

    def upsert(self, collection: str, records: List[VectorRecord]) -> None:
        if not records:
            return
        self.plugin.upsert(collection, records)

    def search(self, collection: str, vector: List[float], limit: int):
        return self.plugin.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )

    def delete_collection(self, collection: str) -> None:
        self.plugin.delete_collection(collection)
