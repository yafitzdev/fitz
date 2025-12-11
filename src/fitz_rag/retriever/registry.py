from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from fitz_rag.retriever.base import RetrievalPlugin
import fitz_rag.retriever.plugins as plugins_pkg

# Global registry populated via auto_discover_plugins().
RETRIEVER_REGISTRY: Dict[str, Type[RetrievalPlugin]] = {}


def auto_discover_plugins() -> None:
    """
    Automatically import all modules in fitz_rag.retriever.plugins,
    find classes that look like retrieval plugins, and register them.

    A class is considered a RetrievalPlugin if:
      - it has a callable `retrieve` attribute
      - it defines a class attribute `plugin_name` (str)
    """
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Duck-typing: any class with `plugin_name` and `retrieve` is treated as a plugin
            plugin_name = getattr(obj, "plugin_name", None)
            retrieve_fn = getattr(obj, "retrieve", None)

            if (
                isinstance(plugin_name, str)
                and callable(retrieve_fn)
            ):
                RETRIEVER_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


# Run auto-discovery once at import time
auto_discover_plugins()


def get_retriever_plugin(name: str) -> Type[RetrievalPlugin]:
    try:
        return RETRIEVER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown retriever plugin: {name!r}") from e
