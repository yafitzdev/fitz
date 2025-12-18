# fitz/llm/registry.py
"""
Central LLM plugin registry - YAML-based system.

Handles discovery and registration of:
- chat plugins (YAML files in fitz/llm/chat/)
- embedding plugins (YAML files in fitz/llm/embedding/)
- rerank plugins (YAML files in fitz/llm/rerank/)

Note: vector_db plugins have their own separate registry at fitz.vector_db.registry

Design principle: NO SILENT FALLBACK
- If user configures "cohere", they get cohere or an error
- Plugins are YAML files, not Python modules
"""
from __future__ import annotations

from typing import Any, List

from fitz.llm.loader import YAMLPluginNotFoundError, list_yaml_plugins
from fitz.llm.runtime import (
    YAMLChatClient,
    YAMLEmbeddingClient,
    YAMLRerankClient,
)

# Import exception classes from core.registry for consistent hierarchy
from fitz.core.registry import (
    PluginRegistryError,
    PluginNotFoundError,
    LLMRegistryError,
)


# =============================================================================
# Registry Implementation
# =============================================================================

# Valid LLM plugin types
VALID_LLM_TYPES = frozenset({"chat", "embedding", "rerank"})


def available_llm_plugins(plugin_type: str) -> List[str]:
    """
    List available LLM plugins for a type.

    Scans YAML files in fitz/llm/{plugin_type}/ directory.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")

    Returns:
        Sorted list of plugin names

    Examples:
        >>> available_llm_plugins("chat")
        ['anthropic', 'cohere', 'openai']
    """
    if plugin_type not in VALID_LLM_TYPES:
        raise ValueError(
            f"Invalid LLM plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(VALID_LLM_TYPES)}"
        )

    return sorted(list_yaml_plugins(plugin_type))


def get_llm_plugin(*, plugin_name: str, plugin_type: str, **kwargs) -> Any:
    """
    Get an LLM plugin instance by name and type.

    Creates a plugin instance from YAML specification.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere", "openai")
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        **kwargs: Plugin initialization parameters (model, temperature, etc.)

    Returns:
        Plugin instance (YAMLChatClient, YAMLEmbeddingClient, or YAMLRerankClient)

    Raises:
        ValueError: If plugin_type is not valid
        LLMRegistryError: If plugin not found

    Examples:
        >>> chat = get_llm_plugin(plugin_name="cohere", plugin_type="chat")
        >>> response = chat.chat([{"role": "user", "content": "Hello"}])

        >>> embedder = get_llm_plugin(plugin_name="openai", plugin_type="embedding")
        >>> vector = embedder.embed("Hello world")
    """
    # Validate plugin_type first
    if plugin_type not in VALID_LLM_TYPES:
        raise ValueError(
            f"Invalid LLM plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(VALID_LLM_TYPES)}"
        )

    # Try to create plugin from YAML
    try:
        if plugin_type == "chat":
            return YAMLChatClient.from_name(plugin_name, **kwargs)
        elif plugin_type == "embedding":
            return YAMLEmbeddingClient.from_name(plugin_name, **kwargs)
        elif plugin_type == "rerank":
            return YAMLRerankClient.from_name(plugin_name, **kwargs)
    except YAMLPluginNotFoundError:
        available = available_llm_plugins(plugin_type)
        raise LLMRegistryError(
            f"Unknown {plugin_type} plugin: {plugin_name!r}. "
            f"Available: {available}"
        )
    except Exception as e:
        raise LLMRegistryError(
            f"Failed to load {plugin_type} plugin '{plugin_name}': {e}"
        ) from e


def resolve_llm_plugin(*, plugin_type: str, requested_name: str, **kwargs) -> Any:
    """
    Alias for get_llm_plugin (backwards compatibility).

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        requested_name: Name of the plugin
        **kwargs: Plugin initialization parameters

    Returns:
        Plugin instance
    """
    return get_llm_plugin(plugin_name=requested_name, plugin_type=plugin_type, **kwargs)


# =============================================================================
# Legacy Registry Objects (for backwards compatibility)
# =============================================================================

# These are kept for backwards compatibility with code that imports them
# They don't do anything - the real registry is the functions above


class _DummyRegistry:
    """Dummy registry object for backwards compatibility."""

    def __init__(self, plugin_type: str):
        self.plugin_type = plugin_type

    def list_available(self) -> List[str]:
        return available_llm_plugins(self.plugin_type)


CHAT_REGISTRY = _DummyRegistry("chat")
EMBEDDING_REGISTRY = _DummyRegistry("embedding")
RERANK_REGISTRY = _DummyRegistry("rerank")

# Combined view for backwards compatibility
LLM_REGISTRY = {
    "chat": CHAT_REGISTRY,
    "embedding": EMBEDDING_REGISTRY,
    "rerank": RERANK_REGISTRY,
}

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Functions
    "get_llm_plugin",
    "available_llm_plugins",
    "resolve_llm_plugin",
    # Registries (legacy)
    "LLM_REGISTRY",
    "CHAT_REGISTRY",
    "EMBEDDING_REGISTRY",
    "RERANK_REGISTRY",
    # Errors (re-exported from core.registry)
    "LLMRegistryError",
    "PluginRegistryError",
    "PluginNotFoundError",
]