# src/fitz_rag/retriever/rag_retriever.py
"""
Embedding-based retriever for fitz-rag.

Supports multiple Qdrant API styles:
- client.query_points(vector=...)               # modern API
- client.query_points(query_vector=...)         # some versions
- client.search(query_vector=...)               # legacy API

This file is intentionally defensive to support multiple Qdrant versions.
"""

from __future__ import annotations

from typing import List, Any, Dict

from qdrant_client import QdrantClient

from fitz_rag.src.core.types import RetrievedChunk
from fitz_rag.src.retriever.base import BaseRetriever


class RAGRetriever(BaseRetriever):
    """
    Standard embedding-based retriever using an embedding client
    and a Qdrant vector store. Works across Qdrant versions.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Any,
        collection: str,
        top_k: int = 10,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        # Compute embedding
        vector = self.embedder.embed(query)

        # -----------------------------------------------------
        # Try modern "query_points" API with vector=...
        # -----------------------------------------------------
        if hasattr(self.client, "query_points"):
            try:
                raw = self.client.query_points(
                    collection_name=self.collection,
                    vector=vector,               # NEW API
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                results = raw.points
                return self._wrap_results(results)
            except TypeError:
                pass

            # -------------------------------------------------
            # Try "query_vector=..." API variant
            # -------------------------------------------------
            try:
                raw = self.client.query_points(
                    collection_name=self.collection,
                    query_vector=vector,         # SOME VERSIONS
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                results = raw.points
                return self._wrap_results(results)
            except TypeError:
                pass

        # -----------------------------------------------------
        # Legacy .search() API
        # -----------------------------------------------------
        if hasattr(self.client, "search"):
            try:
                results = self.client.search(
                    collection_name=self.collection,
                    query_vector=vector,
                    limit=self.top_k,
                    with_payload=True,
                    with_vectors=False,
                )
                return self._wrap_results(results)
            except Exception:
                pass

        # If no path succeeded, error out
        raise RuntimeError(
            "RAGRetriever could not perform search. "
            "Unsupported Qdrant API version."
        )

    # ---------------------------------------------------------
    # Convert Qdrant results → RetrievedChunk objects
    # ---------------------------------------------------------
    def _wrap_results(self, results) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []

        for point in results:
            payload: Dict = point.payload or {}

            chunks.append(
                RetrievedChunk(
                    collection=self.collection,
                    score=float(point.score),
                    text=payload.get("text", ""),
                    metadata=payload,
                    chunk_id=point.id,
                )
            )

        return chunks
