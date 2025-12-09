# src/fitz_rag/llm/rerank_client.py
"""
Rerank client abstractions for fitz-rag.

This module defines:
- RerankClient protocol
- CohereRerankClient: wrapper around Cohere's v2 Rerank API
- DummyRerankClient: deterministic stub useful in tests

You can use these clients directly, or wire them into your retriever /
pipeline objects to implement a two-stage retrieval:

    1) dense vector search in Qdrant -> N candidate chunks
    2) rerank those N candidates with a small cross-encoder (Cohere Rerank)
    3) take top_k for the final context fed into the LLM

Environment variables
---------------------
COHERE_API_KEY          # required for CohereRerankClient (unless api_key passed)
COHERE_RERANK_MODEL     # optional, overrides default model name
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional, Iterable
import os

try:  # Optional dependency
    import cohere
except ImportError:  # pragma: no cover
    cohere = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Rerank protocol
# ---------------------------------------------------------------------------


class RerankClient(Protocol):
    """
    Minimal rerank interface.

    Implementations receive a query + list of document texts and return
    a list of integer indices sorted from most to least relevant.
    """

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        ...


# ---------------------------------------------------------------------------
# Cohere implementation
# ---------------------------------------------------------------------------


@dataclass
class CohereRerankClient:
    """
    Rerank client using Cohere's v2 Rerank API.

    Defaults:
      - model: "rerank-v3.5"  (multilingual, high quality)

    You can override via constructor or COHERE_RERANK_MODEL env var.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError(
                "cohere is not installed. Run `pip install cohere` inside your environment."
            )

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it in your environment or pass api_key=... to CohereRerankClient."
            )

        self.model = (
            self.model
            or os.getenv("COHERE_RERANK_MODEL")
            or "rerank-v3.5"
        )

        self._client = cohere.ClientV2(api_key=key)

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        if not documents:
            return []

        n = top_n if top_n is not None else len(documents)

        res = self._client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=n,
        )

        # Map back to indices in the original list, ordered by relevance_score
        return [r.index for r in res.results]

    # --- Convenience helpers -------------------------------------------------

    def rerank_texts(self, query: str, documents: Iterable[str], top_n: Optional[int] = None) -> List[str]:
        """
        Rerank and return the texts themselves (not the indices).
        """
        docs_list = list(documents)
        order = self.rerank(query, docs_list, top_n=top_n)
        return [docs_list[i] for i in order]


# ---------------------------------------------------------------------------
# Dummy implementation (for tests / offline work)
# ---------------------------------------------------------------------------


@dataclass
class DummyRerankClient:
    """
    Simple deterministic reranker for tests.

    It sorts documents by length descending, pretending that longer
    docs are more relevant.
    """

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:
        if not documents:
            return []

        # Sort indices by length of document (descending)
        indices = list(range(len(documents)))
        indices.sort(key=lambda i: len(documents[i]), reverse=True)

        if top_n is not None:
            indices = indices[:top_n]
        return indices
