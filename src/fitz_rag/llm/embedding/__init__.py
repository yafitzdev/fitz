"""
Embedding subsystem for fitz-rag.

This package exposes:

- EmbeddingPlugin: protocol for all embedding implementations
- EmbeddingEngine: orchestration layer around embedding plugins

Built-in plugins live in:
    fitz_rag.llm.embedding.plugins
"""

from .base import EmbeddingPlugin
from .engine import EmbeddingEngine

__all__ = ["EmbeddingPlugin", "EmbeddingEngine"]
