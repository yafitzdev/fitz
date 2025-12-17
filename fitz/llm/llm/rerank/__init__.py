# ============================
# File: src/fitz_stack/llm/rerank/__init__.py
# ============================
"""
Rerank subsystem for fitz_stack.

This package exposes:
- RerankPlugin: protocol for all rerank implementations
- RerankEngine: orchestration layer around rerank plugins

Plugins live in:
    fitz_stack.llm.rerank.plugins
"""

from .base import RerankPlugin
from .engine import RerankEngine

__all__ = ["RerankPlugin", "RerankEngine"]
