# fitz_sage/llm/__init__.py
"""
LLM provider system for Fitz.

Direct provider wrappers with pluggable authentication. fitz-sage uses
no embeddings, so there is no embedder factory or protocol here.
"""

from __future__ import annotations

# Auth providers
from fitz_sage.llm.auth import ApiKeyAuth, AuthProvider, M2MAuth, NoAuth

# Public API
from fitz_sage.llm.client import get_chat, get_reranker, get_vision
from fitz_sage.llm.factory import ChatFactory, ModelTier, get_chat_factory

# Provider protocols
from fitz_sage.llm.providers.base import (
    ChatProvider,
    RerankProvider,
    RerankResponse,
    RerankResult,
    VisionProvider,
)

__all__ = [
    # Public API
    "get_chat",
    "get_reranker",
    "get_vision",
    # Factory (per-task tier selection)
    "get_chat_factory",
    "ChatFactory",
    "ModelTier",
    # Provider protocols
    "ChatProvider",
    "RerankProvider",
    "VisionProvider",
    "RerankResponse",
    "RerankResult",
    # Auth providers
    "AuthProvider",
    "ApiKeyAuth",
    "M2MAuth",
    "NoAuth",
]
