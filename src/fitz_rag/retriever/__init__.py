"""
Retriever subsystem for fitz-rag.

This package exposes:

- RetrievalPlugin: protocol for all retrieval strategies
- RetrieverEngine: orchestration layer for retrieval plugins

Default dense retrieval implementation lives in:
    fitz_rag.retriever.plugins.dense
"""

from .base import RetrievalPlugin
from .engine import RetrieverEngine

__all__ = ["RetrievalPlugin", "RetrieverEngine"]
