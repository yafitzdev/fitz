from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from core.llm.rerank.base import RerankPlugin
import core.llm.rerank.plugins as plugins_pkg

RERANK_REGISTRY: Dict[str, Type[RerankPlugin]] = {}


def auto_discover_plugins() -> None:
    """
    Automatically import all reranker plugin modules.

    A class is considered a rerank plugin if:
      - it has a class attribute plugin_name (str)
      - it has a callable rerank(query, chunks)
    """
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            plugin_name = getattr(obj, "plugin_name", None)
            rerank_fn = getattr(obj, "rerank", None)

            if isinstance(plugin_name, str) and callable(rerank_fn):
                RERANK_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


# Run auto-discovery at import time
auto_discover_plugins()


def get_rerank_plugin(name: str) -> Type[RerankPlugin]:
    try:
        return RERANK_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown rerank plugin {name!r}") from e
