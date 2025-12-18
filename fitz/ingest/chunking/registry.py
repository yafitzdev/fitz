# fitz/ingest/chunking/registry.py
"""Chunking plugin registry."""
from __future__ import annotations

from typing import Any, List, Type

from fitz.core.registry import CHUNKING_REGISTRY, PluginNotFoundError


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunking plugin by name."""
    return CHUNKING_REGISTRY.get(plugin_name)


def available_chunking_plugins() -> List[str]:
    """List available chunking plugins."""
    return CHUNKING_REGISTRY.list_available()


__all__ = [
    "get_chunking_plugin",
    "available_chunking_plugins",
    "PluginNotFoundError",
]
