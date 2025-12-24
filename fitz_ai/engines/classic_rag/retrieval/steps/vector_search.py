# fitz_ai/engines/classic_rag/retrieval/steps/vector_search.py
"""
Vector Search Step - Initial retrieval from vector database.

Embeds query and searches for top-k candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.exceptions import EmbeddingError, VectorSearchError
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import Embedder, RetrievalStep, VectorClient

logger = get_logger(__name__)


@dataclass
class VectorSearchStep(RetrievalStep):
    """
    Initial vector search step.

    Embeds query and searches vector DB for top-k candidates.
    This is typically the first step that produces initial chunks.

    Args:
        client: Vector database client
        embedder: Embedding service
        collection: Collection name to search
        k: Number of candidates to retrieve (default: 25)
    """

    client: VectorClient
    embedder: Embedder
    collection: str
    k: int = 25  # Retrieve more than final k for downstream filtering

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute vector search.

        Any pre-existing chunks (e.g., artifacts) are preserved and prepended
        to the search results.
        """
        logger.debug(
            f"{RETRIEVER} VectorSearchStep: k={self.k}, collection={self.collection}"
        )

        # 1. Embed query
        try:
            query_vector = self.embedder.embed(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {query!r}") from exc

        # 2. Search vector DB
        try:
            # Try named args first (Qdrant style)
            try:
                hits = self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=self.k,
                    with_payload=True,
                )
            except TypeError:
                # Fall back to positional args
                hits = self.client.search(self.collection, query_vector, self.k)
        except Exception as exc:
            raise VectorSearchError(f"Vector search failed: {exc}") from exc

        # 3. Convert hits to Chunks
        results: list[Chunk] = []
        for idx, hit in enumerate(hits):
            payload = (
                getattr(hit, "payload", None) or getattr(hit, "metadata", None) or {}
            )
            if not isinstance(payload, dict):
                payload = {}

            chunk = Chunk(
                id=str(getattr(hit, "id", idx)),
                doc_id=str(
                    payload.get("doc_id")
                    or payload.get("document_id")
                    or payload.get("source")
                    or "unknown"
                ),
                content=str(payload.get("content") or payload.get("text") or ""),
                chunk_index=int(payload.get("chunk_index", idx)),
                metadata={
                    **payload,
                    "vector_score": getattr(hit, "score", None),
                },
            )
            results.append(chunk)

        logger.debug(f"{RETRIEVER} VectorSearchStep: retrieved {len(results)} chunks")

        # Preserve any pre-existing chunks (e.g., artifacts) by prepending them
        if chunks:
            logger.debug(
                f"{RETRIEVER} VectorSearchStep: preserving {len(chunks)} pre-existing chunks"
            )
            return chunks + results

        return results
