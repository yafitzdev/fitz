# ingest/ingestion/registry.py
from __future__ import annotations

from typing import Dict, Type

from ingest.ingestion.base import IngestPlugin

REGISTRY: Dict[str, Type[IngestPlugin]] = {}


def register(plugin: Type[IngestPlugin]) -> Type[IngestPlugin]:
    name = getattr(plugin, "plugin_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Ingest plugin must define non-empty plugin_name")

    existing = REGISTRY.get(name)
    if existing is not None and existing is not plugin:
        raise ValueError(
            f"Duplicate ingest plugin_name={name!r}: "
            f"{existing.__module__}.{existing.__name__} vs {plugin.__module__}.{plugin.__name__}"
        )

    REGISTRY[name] = plugin
    return plugin


def get_ingest_plugin(name: str) -> Type[IngestPlugin]:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ingester plugin: {name!r}") from exc
