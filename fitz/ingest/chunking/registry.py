# fitz/ingest/chunking/registry.py
"""Chunking plugin registry."""
from __future__ import annotations

from fitz.core.registry import (
    CHUNKING_REGISTRY,
    PluginNotFoundError,
    PluginRegistryError,
    available_chunking_plugins,
    get_chunker_plugin,
    get_chunking_plugin,
)

__all__ = [
    "get_chunker_plugin",
    "get_chunking_plugin",
    "available_chunking_plugins",
    "CHUNKING_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
]
