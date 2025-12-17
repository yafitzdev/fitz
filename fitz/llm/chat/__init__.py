# ============================
# File: src/fitz_stack/llm/chat/__init__.py
# ============================
"""
Chat subsystem for fitz_stack.

This package exposes:
- ChatPlugin: protocol for all chat implementations
- ChatEngine: orchestration layer around chat plugins

Plugins live in:
    fitz_stack.llm.chat.plugins
"""

from .base import ChatPlugin
from .engine import ChatEngine

__all__ = ["ChatPlugin", "ChatEngine"]
