"""
Rerank subsystem for fitz-rag.

This package exposes:
- RerankPlugin: protocol for all rerank implementations
- RerankEngine: orchestration layer for rerank plugins

Built-in plugins live in:
    fitz_rag.llm.rerank.plugins
"""

from .base import RerankPlugin
from .engine import RerankEngine

__all__ = ["RerankPlugin", "RerankEngine"]
