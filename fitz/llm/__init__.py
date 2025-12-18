# fitz/llm/__init__.py
"""
LLM plugin system for Fitz.

This module provides:
- YAML-based plugin definitions for chat, embedding, and rerank
- Runtime clients that execute YAML plugin specs
- Registry integration for plugin discovery

Usage:
    # Using YAML plugins directly
    from fitz.llm.runtime import YAMLChatClient, YAMLEmbeddingClient

    client = YAMLChatClient.from_name("openai", temperature=0.5)
    response = client.chat([{"role": "user", "content": "Hello!"}])

    # Using the registry (discovers all plugins)
    from fitz.llm import get_llm_plugin, available_llm_plugins

    plugins = available_llm_plugins("chat")
    plugin_cls = get_llm_plugin(plugin_name="openai", plugin_type="chat")
"""
from __future__ import annotations

from fitz.llm.registry import available_llm_plugins, get_llm_plugin

# YAML plugin exports
from fitz.llm.loader import (
    load_chat_plugin,
    load_embedding_plugin,
    load_rerank_plugin,
    load_plugin,
    list_yaml_plugins,
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
    # Registry functions (existing)
    "available_llm_plugins",
    "get_llm_plugin",
    # YAML loader functions
    "load_chat_plugin",
    "load_embedding_plugin",
    "load_rerank_plugin",
    "load_plugin",
    "list_yaml_plugins",
    # YAML errors
    "YAMLPluginError",
    "YAMLPluginNotFoundError",
    "YAMLPluginValidationError",
    # YAML runtime clients
    "YAMLChatClient",
    "YAMLEmbeddingClient",
    "YAMLRerankClient",
    # YAML factory functions
    "create_yaml_chat_client",
    "create_yaml_embedding_client",
    "create_yaml_rerank_client",
]