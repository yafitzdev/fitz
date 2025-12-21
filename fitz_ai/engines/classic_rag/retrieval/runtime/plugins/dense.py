# fitz_ai/engines/classic_rag/retrieval/runtime/plugins/dense.py
"""
Dense retrieval plugin using vector search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.exceptions import (
    EmbeddingError,
    RerankError,
    VectorSearchError,
)
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.engines.classic_rag.retrieval.runtime.base import RetrievalPlugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

logger = get_logger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class VectorSearchClient(Protocol):
    def search(self, *args: Any, **kwargs: Any) -> list[Any]: ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding plugins."""

    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerank plugins."""

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]: ...


# =============================================================================
# Config
# =============================================================================


@dataclass(frozen=True, slots=True)
class RetrieverCfg:
    collection: str
    top_k: int = 5


# =============================================================================
# Plugin
# =============================================================================


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    plugin_name: str = "dense"

    client: VectorSearchClient | None = None
    retriever_cfg: RetrieverCfg | None = None

    embedder: Embedder | None = None
    rerank_engine: Reranker | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            raise ValueError("client must be provided")
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")
        if self.embedder is None:
            raise ValueError("embedder must be injected")

    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        try:
            query_vector = self.embedder.embed(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {query!r}") from exc

        try:
            try:
                hits = self.client.search(
                    collection_name=self.retriever_cfg.collection,
                    query_vector=query_vector,
                    limit=self.retriever_cfg.top_k,
                    with_payload=True,
                )
            except TypeError:
                hits = self.client.search(
                    self.retriever_cfg.collection,
                    query_vector,
                    self.retriever_cfg.top_k,
                )
        except Exception as exc:
            raise VectorSearchError("Vector search failed") from exc

        chunks: List[Chunk] = []

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
                    "score": getattr(hit, "score", None),
                },
            )
            chunks.append(chunk)

        if self.rerank_engine and chunks:
            try:
                chunks = self._rerank_chunks(query, chunks)
            except Exception as exc:
                raise RerankError("Reranking failed") from exc

        return chunks

    def _rerank_chunks(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank chunks using the rerank engine.

        The reranker expects a list of strings (document texts), not Chunk objects.
        This method:
        1. Extracts text content from each chunk
        2. Calls the reranker with the text list
        3. Reorders the original Chunk objects based on rerank results

        Args:
            query: The search query
            chunks: List of Chunk objects to rerank

        Returns:
            Reordered list of Chunk objects based on relevance scores
        """
        if not chunks:
            return chunks

        # Extract text content from chunks for the reranker
        documents = [chunk.content for chunk in chunks]

        # Call reranker - returns list of (original_index, score) tuples
        ranked_results = self.rerank_engine.rerank(query, documents)

        # Reorder chunks based on rerank results
        # ranked_results is [(index, score), ...] sorted by relevance
        reranked_chunks: List[Chunk] = []
        for idx, score in ranked_results:
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                # Update metadata with rerank score
                updated_metadata = dict(chunk.metadata)
                updated_metadata["rerank_score"] = score

                reranked_chunk = Chunk(
                    id=chunk.id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    metadata=updated_metadata,
                )
                reranked_chunks.append(reranked_chunk)

        logger.debug(
            f"{RETRIEVER} Reranked {len(chunks)} chunks → {len(reranked_chunks)} results"
        )

        return reranked_chunks
