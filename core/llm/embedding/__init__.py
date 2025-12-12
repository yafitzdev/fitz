# ============================
# File: src/fitz_stack/llm/embedding/__init__.py
# ============================
"""
Embedding subsystem for fitz_stack.

This package exposes:
- EmbeddingPlugin: protocol for all embedding implementations
- EmbeddingEngine: orchestration layer around embedding plugins

Plugins live in:
    fitz_stack.llm.embedding.plugins
"""

from .base import EmbeddingPlugin
from .engine import EmbeddingEngine

__all__ = ["EmbeddingPlugin", "EmbeddingEngine"]
