# fitz/ingest/ingestion/registry.py
"""Ingestion plugin registry."""
from __future__ import annotations

from fitz.core.registry import (
    INGEST_REGISTRY,
    PluginNotFoundError,
    PluginRegistryError,
    available_ingest_plugins,
    get_ingest_plugin,
)

__all__ = [
    "get_ingest_plugin",
    "available_ingest_plugins",
    "INGEST_REGISTRY",
    "PluginRegistryError",
    "PluginNotFoundError",
]
