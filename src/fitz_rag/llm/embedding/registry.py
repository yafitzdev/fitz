from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from fitz_rag.llm.embedding.base import EmbeddingPlugin
import fitz_rag.llm.embedding.plugins as plugins_pkg

# Global registry populated via auto_discover_plugins().
EMBEDDING_REGISTRY: Dict[str, Type[EmbeddingPlugin]] = {}


def auto_discover_plugins() -> None:
    """
    Automatically import all modules in fitz_rag.llm.embedding.plugins,
    find classes that look like embedding plugins, and register them.

    A class is considered an EmbeddingPlugin if:
      - it has a class attribute `plugin_name` (str)
      - it has a callable `embed` method

    This is duck-typed; we do not require explicit inheritance from a
    base class, which makes it friendly for LLM-generated plugins.
    """
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            plugin_name = getattr(obj, "plugin_name", None)
            embed_fn = getattr(obj, "embed", None)

            if isinstance(plugin_name, str) and callable(embed_fn):
                EMBEDDING_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


# Run auto-discovery once at import time
auto_discover_plugins()


def get_embedding_plugin(name: str) -> Type[EmbeddingPlugin]:
    try:
        return EMBEDDING_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown embedding plugin: {name!r}") from e
