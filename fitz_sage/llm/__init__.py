# fitz_sage/llm/__init__.py
"""
LLM provider system for Fitz.

Direct provider wrappers with pluggable authentication.
"""

from __future__ import annotations

# Auth providers
from fitz_sage.llm.auth import ApiKeyAuth, AuthProvider, M2MAuth, NoAuth

# Public API
from fitz_sage.llm.client import get_chat, get_embedder, get_reranker, get_vision
from fitz_sage.llm.factory import ChatFactory, ModelTier, get_chat_factory

# Provider protocols
from fitz_sage.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    RerankProvider,
    RerankResult,
    VisionProvider,
)

__all__ = [
    # Public API
    "get_chat",
    "get_embedder",
    "get_reranker",
    "get_vision",
    # Factory (per-task tier selection)
    "get_chat_factory",
    "ChatFactory",
    "ModelTier",
    # Provider protocols
    "ChatProvider",
    "EmbeddingProvider",
    "RerankProvider",
    "VisionProvider",
    "RerankResult",
    # Auth providers
    "AuthProvider",
    "ApiKeyAuth",
    "M2MAuth",
    "NoAuth",
]
