# ingest/chunking/registry.py

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from ingest.chunking.base import BaseChunker
import ingest.chunking.plugins as plugins_pkg

CHUNKER_REGISTRY: Dict[str, Type[BaseChunker]] = {}


def auto_discover_plugins() -> None:
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            plugin_name = getattr(obj, "plugin_name", None)
            chunk_fn = getattr(obj, "chunk_text", None)

            if isinstance(plugin_name, str) and callable(chunk_fn):
                CHUNKER_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


auto_discover_plugins()


def get_chunker_plugin(name: str) -> Type[BaseChunker]:
    try:
        return CHUNKER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown chunker plugin: {name!r}") from e
