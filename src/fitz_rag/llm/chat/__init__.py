"""
Chat / LLM subsystem for fitz-rag.

This package exposes:
- ChatPlugin: protocol for all chat implementations
- ChatEngine: orchestration layer around chat plugins

Built-in plugins live in:
    fitz_rag.llm.chat.plugins
"""

from .base import ChatPlugin
from .engine import ChatEngine

__all__ = ["ChatPlugin", "ChatEngine"]
