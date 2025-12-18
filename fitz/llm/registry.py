# fitz/llm/registry.py
"""
LLM plugin registry - YAML-based system.

Handles discovery and instantiation of:
- chat plugins (YAML files in fitz/llm/chat/)
- embedding plugins (YAML files in fitz/llm/embedding/)
- rerank plugins (YAML files in fitz/llm/rerank/)

Design principle: NO SILENT FALLBACK
- If user configures "cohere", they get cohere or an error
- Plugins are YAML files, not Python modules
"""
from __future__ import annotations

from typing import Any, List

from fitz.core.registry import LLMRegistryError
from fitz.llm.loader import YAMLPluginNotFoundError, list_yaml_plugins
from fitz.llm.runtime import create_yaml_client

VALID_LLM_TYPES = frozenset({"chat", "embedding", "rerank"})


def available_llm_plugins(plugin_type: str) -> List[str]:
    """
    List available LLM plugins for a type.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")

    Returns:
        Sorted list of plugin names
    """
    if plugin_type not in VALID_LLM_TYPES:
        raise ValueError(
            f"Invalid LLM plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(VALID_LLM_TYPES)}"
        )

    return sorted(list_yaml_plugins(plugin_type))


def get_llm_plugin(*, plugin_name: str, plugin_type: str, **kwargs: Any) -> Any:
    """
    Get an LLM plugin instance by name and type.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere", "openai")
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        **kwargs: Plugin initialization parameters

    Returns:
        Plugin instance

    Raises:
        ValueError: If plugin_type is invalid
        LLMRegistryError: If plugin not found
    """
    if plugin_type not in VALID_LLM_TYPES:
        raise ValueError(
            f"Invalid LLM plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(VALID_LLM_TYPES)}"
        )

    try:
        return create_yaml_client(plugin_type, plugin_name, **kwargs)
    except YAMLPluginNotFoundError:
        available = available_llm_plugins(plugin_type)
        raise LLMRegistryError(
            f"Unknown {plugin_type} plugin: {plugin_name!r}. " f"Available: {available}"
        )
    except Exception as e:
        raise LLMRegistryError(f"Failed to load {plugin_type} plugin '{plugin_name}': {e}") from e


__all__ = [
    "get_llm_plugin",
    "available_llm_plugins",
    "LLMRegistryError",
]
