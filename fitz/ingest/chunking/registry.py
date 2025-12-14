# ingest/chunking/registry.py
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Iterable, Type

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CHUNKING
from fitz.ingest.chunking.base import ChunkerPlugin

logger = get_logger(__name__)

CHUNKER_REGISTRY: Dict[str, Type[ChunkerPlugin]] = {}
_DISCOVERED = False


def _iter_plugin_classes(module: object) -> Iterable[type]:
    for obj in vars(module).values():
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != getattr(module, "__name__", None):
            continue

        plugin_name = getattr(obj, "plugin_name", None)
        if not isinstance(plugin_name, str) or not plugin_name:
            continue

        chunk_fn = getattr(obj, "chunk_text", None)
        if not callable(chunk_fn):
            continue

        yield obj


def _auto_discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    plugins_pkg = importlib.import_module("fitz.ingest.chunking.plugins")

    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(f"{plugins_pkg.__name__}.{module_info.name}")

        for cls in _iter_plugin_classes(module):
            name = getattr(cls, "plugin_name")
            existing = CHUNKER_REGISTRY.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Duplicate chunker plugin_name={name!r}: "
                    f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
                )
            CHUNKER_REGISTRY[name] = cls  # type: ignore[assignment]

    logger.info(f"{CHUNKING} Discovered chunker plugins: {sorted(CHUNKER_REGISTRY.keys())}")
    _DISCOVERED = True


def get_chunker_plugin(name: str) -> Type[ChunkerPlugin]:
    _auto_discover()
    try:
        return CHUNKER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown chunker plugin: {name!r}") from e
