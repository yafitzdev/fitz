# rag/retrieval/registry.py
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Type

from rag.retrieval.base import RetrievalPlugin


RETRIEVER_REGISTRY: Dict[str, Type[RetrievalPlugin]] = {}
_DISCOVERED = False


def _auto_discover() -> None:
    """
    Deterministic plugin discovery.

    Architecture contract:
    - The plugins package must NOT import the registry (avoid circular imports).
    - Registry owns discovery and registration.
    - Plugins expose a `plugin_name: str` class attribute.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    plugins_pkg = importlib.import_module("rag.retrieval.plugins")

    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(f"{plugins_pkg.__name__}.{module_info.name}")

        for obj in vars(module).values():
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, RetrievalPlugin) or obj is RetrievalPlugin:
                continue

            plugin_name = getattr(obj, "plugin_name", None)
            if isinstance(plugin_name, str) and plugin_name:
                RETRIEVER_REGISTRY[plugin_name] = obj

    _DISCOVERED = True


def get_retriever_plugin(name: str) -> Type[RetrievalPlugin]:
    _auto_discover()
    try:
        return RETRIEVER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown retriever plugin: {name!r}") from e
