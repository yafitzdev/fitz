# ============================
# File: src/fitz_rag/retriever/registry.py
# ============================
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from rag.retriever.base import RetrievalPlugin
import rag.retriever.plugins as plugins_pkg


RETRIEVER_REGISTRY: Dict[str, Type[RetrievalPlugin]] = {}


def auto_discover_plugins() -> None:
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module = importlib.import_module(f"{plugins_pkg.__name__}.{module_info.name}")

        for _, obj in inspect.getmembers(module, inspect.isclass):
            plugin_name = getattr(obj, "plugin_name", None)
            retrieve_fn = getattr(obj, "retrieve", None)

            if isinstance(plugin_name, str) and callable(retrieve_fn):
                RETRIEVER_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


auto_discover_plugins()


def get_retriever_plugin(name: str) -> Type[RetrievalPlugin]:
    try:
        return RETRIEVER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown retriever plugin: {name!r}") from e
