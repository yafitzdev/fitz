# fitz_ai/engines/classic_rag/retrieval/runtime/registry.py
"""Retriever plugin registry."""

from __future__ import annotations

from typing import Any, List, Type

from fitz_ai.core.registry import RETRIEVER_REGISTRY, PluginNotFoundError


def get_retriever_plugin(plugin_name: str) -> Type[Any]:
    """Get a retriever plugin by name."""
    return RETRIEVER_REGISTRY.get(plugin_name)


def available_retriever_plugins() -> List[str]:
    """List available retriever plugins."""
    return RETRIEVER_REGISTRY.list_available()


__all__ = [
    "get_retriever_plugin",
    "available_retriever_plugins",
    "PluginNotFoundError",
]
