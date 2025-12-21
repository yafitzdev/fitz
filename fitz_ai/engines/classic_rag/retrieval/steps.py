# fitz_ai/engines/classic_rag/retrieval/steps.py
"""
Retrieval Steps - Standard composable building blocks.

These steps are always the same Python logic. Users orchestrate them
via YAML config by specifying which steps to run and their parameters.

Pipeline: vector_search(k=25) → rerank(k=10) → threshold(τ) → limit(k=5)

Each step:
- Takes a query + list of chunks (or nothing for initial step)
- Returns a list of chunks
- Is stateless and reusable
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.exceptions import (
    EmbeddingError,
    RerankError,
    VectorSearchError,
)
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# =============================================================================
# Protocols for Dependencies
# =============================================================================


@runtime_checkable
class VectorClient(Protocol):
    """Protocol for vector database clients."""

    def search(self, *args: Any, **kwargs: Any) -> list[Any]: ...


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding services."""

    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking services."""

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]: ...


# =============================================================================
# Base Step
# =============================================================================


@dataclass
class RetrievalStep(ABC):
    """Base class for retrieval steps."""

    @abstractmethod
    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Execute step and return updated chunks."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# =============================================================================
# Step: Vector Search (Initial Retrieval)
# =============================================================================


@dataclass
class VectorSearchStep(RetrievalStep):
    """
    Initial vector search step.

    Embeds query and searches vector DB for top-k candidates.
    This is typically the first step that produces initial chunks.
    """

    client: VectorClient
    embedder: Embedder
    collection: str
    k: int = 25  # Retrieve more than final k for downstream filtering

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Execute vector search.

        Note: `chunks` parameter is ignored - this step produces initial chunks.
        """
        logger.debug(f"{RETRIEVER} VectorSearchStep: k={self.k}, collection={self.collection}")

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
            payload = getattr(hit, "payload", None) or getattr(hit, "metadata", None) or {}
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
        return results


# =============================================================================
# Step: Rerank
# =============================================================================


@dataclass
class RerankStep(RetrievalStep):
    """
    Rerank chunks using a cross-encoder or similar model.

    Takes top-k chunks from previous step, reranks them, returns top rerank_k.
    """

    reranker: Reranker
    k: int = 10  # Return top k after reranking

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(f"{RETRIEVER} RerankStep: input={len(chunks)}, k={self.k}")

        # Extract text for reranker
        documents = [chunk.content for chunk in chunks]

        try:
            # Reranker returns [(index, score), ...] sorted by relevance
            ranked_results = self.reranker.rerank(query, documents, top_n=self.k)
        except Exception as exc:
            raise RerankError(f"Reranking failed: {exc}") from exc

        # Reorder chunks based on rerank results
        reranked: list[Chunk] = []
        for idx, score in ranked_results:
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                # Add rerank score to metadata
                updated_metadata = dict(chunk.metadata)
                updated_metadata["rerank_score"] = score

                reranked.append(
                    Chunk(
                        id=chunk.id,
                        doc_id=chunk.doc_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        metadata=updated_metadata,
                    )
                )

        logger.debug(f"{RETRIEVER} RerankStep: output={len(reranked)} chunks")
        return reranked


# =============================================================================
# Step: Threshold Filter
# =============================================================================


@dataclass
class ThresholdStep(RetrievalStep):
    """
    Filter chunks by score threshold.

    Removes chunks below the threshold. Uses rerank_score if available,
    otherwise falls back to vector_score.
    """

    threshold: float = 0.5
    score_key: str = "rerank_score"  # Which score to use
    fallback_key: str = "vector_score"  # Fallback if primary not found

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(f"{RETRIEVER} ThresholdStep: τ={self.threshold}, input={len(chunks)}")

        filtered: list[Chunk] = []
        for chunk in chunks:
            score = chunk.metadata.get(self.score_key)
            if score is None:
                score = chunk.metadata.get(self.fallback_key)
            if score is None:
                # No score - include by default
                filtered.append(chunk)
                continue

            if score >= self.threshold:
                filtered.append(chunk)

        logger.debug(f"{RETRIEVER} ThresholdStep: output={len(filtered)} chunks")
        return filtered


# =============================================================================
# Step: Limit (Final K)
# =============================================================================


@dataclass
class LimitStep(RetrievalStep):
    """
    Limit output to final k chunks.

    Simple truncation - assumes chunks are already sorted by relevance.
    """

    k: int = 5

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        logger.debug(f"{RETRIEVER} LimitStep: k={self.k}, input={len(chunks)}")
        return chunks[: self.k]


# =============================================================================
# Step: Dedupe
# =============================================================================


@dataclass
class DedupeStep(RetrievalStep):
    """
    Remove duplicate chunks based on content.

    Keeps the first occurrence (assumes sorted by relevance).
    """

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(f"{RETRIEVER} DedupeStep: input={len(chunks)}")

        seen: set[str] = set()
        unique: list[Chunk] = []

        for chunk in chunks:
            # Normalize content for comparison
            key = chunk.content.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)

        logger.debug(f"{RETRIEVER} DedupeStep: output={len(unique)} chunks")
        return unique


# =============================================================================
# Step Registry
# =============================================================================

STEP_REGISTRY: dict[str, type[RetrievalStep]] = {
    "vector_search": VectorSearchStep,
    "rerank": RerankStep,
    "threshold": ThresholdStep,
    "limit": LimitStep,
    "dedupe": DedupeStep,
}


def get_step_class(step_type: str) -> type[RetrievalStep]:
    """Get step class by type name."""
    if step_type not in STEP_REGISTRY:
        available = list(STEP_REGISTRY.keys())
        raise ValueError(f"Unknown step type: {step_type!r}. Available: {available}")
    return STEP_REGISTRY[step_type]


def list_available_steps() -> list[str]:
    """List all available step types."""
    return list(STEP_REGISTRY.keys())