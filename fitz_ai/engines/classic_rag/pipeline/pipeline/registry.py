# fitz_ai/engines/classic_rag/pipeline/pipeline/registry.py
"""Pipeline plugin registry."""

from __future__ import annotations

from typing import Any, List, Type

from fitz_ai.core.registry import PIPELINE_REGISTRY, PluginNotFoundError


def get_pipeline_plugin(plugin_name: str) -> Type[Any]:
    """Get a pipeline plugin by name."""
    return PIPELINE_REGISTRY.get(plugin_name)


def available_pipeline_plugins() -> List[str]:
    """List available pipeline plugins."""
    return PIPELINE_REGISTRY.list_available()


__all__ = [
    "get_pipeline_plugin",
    "available_pipeline_plugins",
    "PluginNotFoundError",
]
