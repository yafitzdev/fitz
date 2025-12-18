# fitz/llm/registry.py
"""
Central LLM plugin registry.

Handles discovery and registration of:
- chat plugins
- embedding plugins
- rerank plugins

Note: vector_db plugins have their own separate registry at fitz.vector_db.registry

Design principle: NO SILENT FALLBACK
- If user configures "cohere", they get cohere or an error
- If user wants local, they explicitly configure "local"
- No magic substitution that could cause confusion
"""
from __future__ import annotations

from fitz.core.registry import (  # Functions; Registries; Errors
    CHAT_REGISTRY,
    EMBEDDING_REGISTRY,
    LLM_REGISTRY,
    RERANK_REGISTRY,
    LLMRegistryError,
    PluginNotFoundError,
    PluginRegistryError,
    available_llm_plugins,
    get_llm_plugin,
    resolve_llm_plugin,
)

# Type alias for backwards compat
LLMPluginType = str  # "chat" | "embedding" | "rerank"

__all__ = [
    # Functions
    "get_llm_plugin",
    "available_llm_plugins",
    "resolve_llm_plugin",
    # Registries
    "LLM_REGISTRY",
    "CHAT_REGISTRY",
    "EMBEDDING_REGISTRY",
    "RERANK_REGISTRY",
    # Errors
    "LLMRegistryError",
    "PluginRegistryError",
    "PluginNotFoundError",
    # Types
    "LLMPluginType",
]