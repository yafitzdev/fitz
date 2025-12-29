# fitz_ai/ingestion/ingestion/registry.py
"""Ingestion plugin registry."""

from __future__ import annotations

from typing import Any, List, Type

from fitz_ai.core.registry import INGEST_REGISTRY, PluginNotFoundError


def get_ingest_plugin(plugin_name: str) -> Type[Any]:
    """Get an ingestion plugin by name."""
    return INGEST_REGISTRY.get(plugin_name)


def available_ingest_plugins() -> List[str]:
    """List available ingestion plugins."""
    return INGEST_REGISTRY.list_available()


__all__ = [
    "get_ingest_plugin",
    "available_ingest_plugins",
    "PluginNotFoundError",
]
