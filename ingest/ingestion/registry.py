# ingest/ingestion/registry.py

from __future__ import annotations

from typing import Dict, Type
from ingest.ingestion.base import IngestPlugin

REGISTRY: Dict[str, Type[IngestPlugin]] = {}


def register(plugin: Type[IngestPlugin]) -> Type[IngestPlugin]:
    REGISTRY[plugin.plugin_name] = plugin
    return plugin


def get_ingest_plugin(name: str) -> Type[IngestPlugin]:
    if name not in REGISTRY:
        raise ValueError(f"Unknown ingester plugin: {name}")
    return REGISTRY[name]
