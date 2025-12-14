"""
RAG-level retrieval orchestration API.

This module wires RAG to the core retrieval system.
"""

from fitz.retrieval.runtime.base import RetrievalPlugin

__all__ = [
    "RetrievalPlugin",
    "RetrieverEngine",
]