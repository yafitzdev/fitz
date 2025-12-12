from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from ingest.chunker.base import ChunkerPlugin, BaseChunker
import ingest.chunker.plugins as plugins_pkg

# Global registry of discovered chunker plugins
CHUNKER_REGISTRY: Dict[str, Type[BaseChunker]] = {}


def auto_discover_plugins() -> None:
    """
    Automatically import all modules in fitz_ingest.chunker.plugins
    and register any class that looks like a chunker plugin.

    A class is considered a chunker plugin if:
      - it has a class attribute `plugin_name` (str)
      - it has a callable `chunk_text(self, text, base_meta)`
    """
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            plugin_name = getattr(obj, "plugin_name", None)
            chunk_fn = getattr(obj, "chunk_text", None)

            if isinstance(plugin_name, str) and callable(chunk_fn):
                # We store it as BaseChunker for convenience, but this is
                # duck-typed: subclasses of BaseChunker or standalone
                # classes with the right shape are accepted.
                CHUNKER_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


# Run discovery at import time
auto_discover_plugins()


def get_chunker_plugin(name: str) -> Type[BaseChunker]:
    """
    Look up a chunker plugin class by its registered name.
    """
    try:
        return CHUNKER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown chunker plugin: {name!r}") from e
