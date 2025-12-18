# fitz/llm/__init__.py
"""
LLM plugin system for Fitz.

All LLM plugins are YAML-based.
"""
from __future__ import annotations

from fitz.llm.registry import available_llm_plugins, get_llm_plugin

from fitz.llm.loader import (
    load_chat_plugin,
    load_embedding_plugin,
    load_rerank_plugin,
    load_plugin,
    list_yaml_plugins,
    clear_cache,
    YAMLPluginError,
    YAMLPluginNotFoundError,
    YAMLPluginValidationError,
)

from fitz.llm.runtime import (
    YAMLChatClient,
    YAMLEmbeddingClient,
    YAMLRerankClient,
    create_yaml_chat_client,
    create_yaml_embedding_client,
    create_yaml_rerank_client,
)

__all__ = [
    # Registry
    "available_llm_plugins",
    "get_llm_plugin",
    # Loader
    "load_chat_plugin",
    "load_embedding_plugin",
    "load_rerank_plugin",
    "load_plugin",
    "list_yaml_plugins",
    "clear_cache",
    # Errors
    "YAMLPluginError",
    "YAMLPluginNotFoundError",
    "YAMLPluginValidationError",
    # Runtime clients
    "YAMLChatClient",
    "YAMLEmbeddingClient",
    "YAMLRerankClient",
    # Factory functions
    "create_yaml_chat_client",
    "create_yaml_embedding_client",
    "create_yaml_rerank_client",
]