# fitz/engines/classic_rag/pipeline/pipeline/registry.py
"""Pipeline plugin registry."""
from __future__ import annotations

from fitz.core.registry import (
    PIPELINE_REGISTRY,
    PluginNotFoundError,
    PluginRegistryError,
    available_pipeline_plugins,
    get_pipeline_plugin,
)

__all__ = [
    "get_pipeline_plugin",
    "available_pipeline_plugins",
    "PIPELINE_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
]
