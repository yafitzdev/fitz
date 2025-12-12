from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Type

from core.llm.chat.base import ChatPlugin
import core.llm.chat.plugins as plugins_pkg

CHAT_REGISTRY: Dict[str, Type[ChatPlugin]] = {}


def auto_discover_plugins() -> None:
    """
    Automatically import all chat plugin modules.

    A class is considered a ChatPlugin if:
      - it has a class attribute plugin_name (str)
      - it has a callable chat(messages) method
    """
    package_path = plugins_pkg.__path__

    for module_info in pkgutil.iter_modules(package_path):
        module_name = f"{plugins_pkg.__name__}.{module_info.name}"
        module = importlib.import_module(module_name)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            plugin_name = getattr(obj, "plugin_name", None)
            chat_fn = getattr(obj, "chat", None)

            if isinstance(plugin_name, str) and callable(chat_fn):
                CHAT_REGISTRY[plugin_name] = obj  # type: ignore[assignment]


# Run auto-discovery at import time
auto_discover_plugins()


def get_chat_plugin(name: str) -> Type[ChatPlugin]:
    try:
        return CHAT_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown chat plugin {name!r}") from e
