# src/fitz_rag/retriever/base.py
"""
Base Retriever interface for fitz_rag.

This defines the minimal contract for all retrievers:
- Input: query string
- Output: list[RetrievedChunk]

Concrete implementations include:
- RAGRetriever (embedding-based retrieval)
- Deterministic strategies that do no embedding (via plugins)
"""

from __future__ import annotations

from typing import List, Protocol

from fitz_rag.core import RetrievedChunk


class BaseRetriever(Protocol):
    """
    Basic retrieval protocol.

    Any retriever used by fitz_rag must implement `retrieve(query: str)`.
    """

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        ...


__all__ = ["BaseRetriever"]
