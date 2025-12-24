# fitz_ai/engines/classic_rag/retrieval/steps/base.py
"""
Base classes and protocols for retrieval steps.

All retrieval steps inherit from RetrievalStep and implement execute().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.models.chunk import Chunk

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
    """
    Base class for retrieval steps.

    All steps take a query and list of chunks, and return an updated list of chunks.
    Steps are stateless and composable.
    """

    @abstractmethod
    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Execute step and return updated chunks."""
        ...

    @property
    def name(self) -> str:
        """Return the step class name."""
        return self.__class__.__name__
