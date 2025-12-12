# rag/retrieval/registry.py

from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Iterable, Type

from rag.retrieval.base import RetrievalPlugin

from core.logging.logger import get_logger
from core.logging.tags import RETRIEVER

logger = get_logger(__name__)

RETRIEVER_REGISTRY: Dict[str, Type[RetrievalPlugin]] = {}
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

        retrieve_fn = getattr(obj, "retrieve", None)
        if not callable(retrieve_fn):
            continue

        yield obj


def _auto_discover() -> None:
    """
    Deterministic plugin discovery.

    Architecture contract:
    - `rag.retrieval.plugins.__init__` must be import-free (no registry imports).
    - Registry owns discovery and registration.
    - Plugin classes expose `plugin_name: str` and implement `retrieve(...)`.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return

    plugins_pkg = importlib.import_module("rag.retrieval.plugins")

    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(f"{plugins_pkg.__name__}.{module_info.name}")

        for cls in _iter_plugin_classes(module):
            name = getattr(cls, "plugin_name")
            existing = RETRIEVER_REGISTRY.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Duplicate retriever plugin_name={name!r}: "
                    f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
                )

            RETRIEVER_REGISTRY[name] = cls  # type: ignore[assignment]

    logger.info(f"{RETRIEVER} Discovered retriever plugins: {sorted(RETRIEVER_REGISTRY.keys())}")
    _DISCOVERED = True


def get_retriever_plugin(name: str) -> Type[RetrievalPlugin]:
    _auto_discover()
    try:
        return RETRIEVER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown retriever plugin: {name!r}") from e
