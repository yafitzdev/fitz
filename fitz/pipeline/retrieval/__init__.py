"""
RAG-level retrieval orchestration API.

This module wires RAG to the core retrieval system.
"""

from fitz.retrieval.base import RetrievalPlugin
from fitz.retrieval.engine import RetrieverEngine

__all__ = [
    "RetrievalPlugin",
    "RetrieverEngine",
]