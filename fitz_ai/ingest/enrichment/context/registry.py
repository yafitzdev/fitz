# fitz_ai/ingest/enrichment/context/registry.py
"""
Context builder plugin registry with auto-discovery.

Discovers and manages context builder plugins from the plugins/ directory.
Each plugin must define:
    - plugin_name: str
    - plugin_type: str = "context"
    - supported_extensions: set[str]
    - Builder class implementing ContextBuilder protocol
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from fitz_ai.ingest.enrichment.base import ContextBuilder, EnrichmentContext

logger = logging.getLogger(__name__)


class ContextPluginInfo:
    """Information about a discovered context builder plugin."""

    def __init__(
        self,
        name: str,
        module: Any,
        builder_class: Type,
        supported_extensions: Set[str],
    ):
        self.name = name
        self.module = module
        self.builder_class = builder_class
        self.supported_extensions = supported_extensions

    def create_builder(self, **kwargs) -> Any:
        """Create a builder instance."""
        return self.builder_class(**kwargs)


class ContextRegistry:
    """
    Registry for context builder plugins.

    Auto-discovers plugins from the plugins/ directory and provides
    access to builders filtered by file extension.

    Usage:
        registry = ContextRegistry()
        registry.discover()

        # Get builder for a file
        builder = registry.get_builder_for_extension(".py")
        if builder:
            context = builder.build(file_path, content)
    """

    _instance: "ContextRegistry | None" = None

    def __init__(self):
        self._plugins: Dict[str, ContextPluginInfo] = {}
        self._extension_map: Dict[str, ContextPluginInfo] = {}
        self._discovered = False

    @classmethod
    def get_instance(cls) -> "ContextRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.discover()
        return cls._instance

    def discover(self) -> None:
        """Discover all context builder plugins in the plugins/ directory."""
        if self._discovered:
            return

        plugins_dir = Path(__file__).parent / "plugins"
        if not plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return

        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                self._load_plugin(plugin_file)
            except Exception as e:
                logger.warning(f"Failed to load plugin {plugin_file.name}: {e}")

        self._discovered = True
        logger.info(f"Discovered {len(self._plugins)} context builder plugins")

    def _load_plugin(self, plugin_file: Path) -> None:
        """Load a single plugin from a file."""
        module_name = f"fitz_ai.ingest.enrichment.context.plugins.{plugin_file.stem}"
        module = importlib.import_module(module_name)

        if not hasattr(module, "plugin_name"):
            logger.debug(f"Skipping {plugin_file.name}: no plugin_name")
            return

        if not hasattr(module, "plugin_type") or module.plugin_type != "context":
            logger.debug(f"Skipping {plugin_file.name}: not a context plugin")
            return

        if not hasattr(module, "Builder"):
            logger.debug(f"Skipping {plugin_file.name}: no Builder class")
            return

        plugin_info = ContextPluginInfo(
            name=module.plugin_name,
            module=module,
            builder_class=module.Builder,
            supported_extensions=getattr(module, "supported_extensions", set()),
        )

        self._plugins[plugin_info.name] = plugin_info

        # Map extensions to this plugin
        for ext in plugin_info.supported_extensions:
            ext_lower = ext.lower()
            if ext_lower in self._extension_map:
                logger.debug(
                    f"Extension {ext_lower} already mapped, overwriting with {plugin_info.name}"
                )
            self._extension_map[ext_lower] = plugin_info

        logger.debug(f"Loaded context plugin: {plugin_info.name}")

    def get_plugin(self, name: str) -> ContextPluginInfo | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_plugin_for_extension(self, extension: str) -> ContextPluginInfo | None:
        """Get the plugin that handles a specific file extension."""
        return self._extension_map.get(extension.lower())

    def get_all_plugins(self) -> List[ContextPluginInfo]:
        """Get all discovered plugins."""
        return list(self._plugins.values())

    def list_plugin_names(self) -> List[str]:
        """List all plugin names."""
        return list(self._plugins.keys())

    def list_supported_extensions(self) -> Set[str]:
        """List all supported file extensions."""
        return set(self._extension_map.keys())


def get_context_registry() -> ContextRegistry:
    """Get the context builder registry singleton."""
    return ContextRegistry.get_instance()


def get_context_plugin(name: str) -> ContextPluginInfo | None:
    """Get a context builder plugin by name."""
    return get_context_registry().get_plugin(name)


def list_context_plugins() -> List[str]:
    """List all available context builder plugins."""
    return get_context_registry().list_plugin_names()


__all__ = [
    "ContextRegistry",
    "ContextPluginInfo",
    "get_context_registry",
    "get_context_plugin",
    "list_context_plugins",
]
