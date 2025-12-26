# fitz_ai/ingest/enrichment/artifacts/registry.py
"""
Artifact plugin registry with auto-discovery.

Discovers and manages artifact generator plugins from the plugins/ directory.
Each plugin must define:
    - plugin_name: str
    - plugin_type: str = "artifact"
    - supported_types: set[ContentType]
    - requires_llm: bool
    - Generator class
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, Set, Type, runtime_checkable

from fitz_ai.ingest.enrichment.base import ContentType

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class ArtifactPluginInfo:
    """Information about a discovered artifact plugin."""

    def __init__(
        self,
        name: str,
        module: Any,
        generator_class: Type,
        supported_types: Set[ContentType],
        requires_llm: bool,
        description: str = "",
    ):
        self.name = name
        self.module = module
        self.generator_class = generator_class
        self.supported_types = supported_types
        self.requires_llm = requires_llm
        self.description = description

    def create_generator(self, chat_client: ChatClient | None = None) -> Any:
        """Create a generator instance."""
        if self.requires_llm:
            if chat_client is None:
                raise ValueError(
                    f"Plugin '{self.name}' requires an LLM client but none provided"
                )
            return self.generator_class(chat_client)
        return self.generator_class()


class ArtifactRegistry:
    """
    Registry for artifact generator plugins.

    Auto-discovers plugins from the plugins/ directory and provides
    access to generators filtered by content type.

    Usage:
        registry = ArtifactRegistry()
        registry.discover()

        # Get all plugins applicable to code
        plugins = registry.get_plugins_for_type(ContentType.PYTHON)

        # Generate artifacts
        for plugin in plugins:
            generator = plugin.create_generator(chat_client)
            artifact = generator.generate(analysis)
    """

    _instance: "ArtifactRegistry | None" = None

    def __init__(self):
        self._plugins: Dict[str, ArtifactPluginInfo] = {}
        self._discovered = False

    @classmethod
    def get_instance(cls) -> "ArtifactRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.discover()
        return cls._instance

    def discover(self) -> None:
        """Discover all artifact plugins in the plugins/ directory."""
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
        logger.info(f"Discovered {len(self._plugins)} artifact plugins")

    def _load_plugin(self, plugin_file: Path) -> None:
        """Load a single plugin from a file."""
        module_name = f"fitz_ai.ingest.enrichment.artifacts.plugins.{plugin_file.stem}"
        module = importlib.import_module(module_name)

        if not hasattr(module, "plugin_name"):
            logger.debug(f"Skipping {plugin_file.name}: no plugin_name")
            return

        if not hasattr(module, "plugin_type") or module.plugin_type != "artifact":
            logger.debug(f"Skipping {plugin_file.name}: not an artifact plugin")
            return

        if not hasattr(module, "Generator"):
            logger.debug(f"Skipping {plugin_file.name}: no Generator class")
            return

        plugin_info = ArtifactPluginInfo(
            name=module.plugin_name,
            module=module,
            generator_class=module.Generator,
            supported_types=getattr(module, "supported_types", set()),
            requires_llm=getattr(module, "requires_llm", False),
            description=getattr(module, "description", ""),
        )

        self._plugins[plugin_info.name] = plugin_info
        logger.debug(f"Loaded artifact plugin: {plugin_info.name}")

    def get_plugin(self, name: str) -> ArtifactPluginInfo | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def get_all_plugins(self) -> List[ArtifactPluginInfo]:
        """Get all discovered plugins."""
        return list(self._plugins.values())

    def get_plugins_for_type(
        self, content_type: ContentType
    ) -> List[ArtifactPluginInfo]:
        """Get plugins that support a specific content type."""
        return [
            plugin
            for plugin in self._plugins.values()
            if content_type in plugin.supported_types
        ]

    def get_plugins_by_names(self, names: List[str]) -> List[ArtifactPluginInfo]:
        """Get specific plugins by name."""
        return [self._plugins[name] for name in names if name in self._plugins]

    def list_plugin_names(self) -> List[str]:
        """List all plugin names."""
        return list(self._plugins.keys())


def get_artifact_registry() -> ArtifactRegistry:
    """Get the artifact registry singleton."""
    return ArtifactRegistry.get_instance()


def get_artifact_plugin(name: str) -> ArtifactPluginInfo | None:
    """Get an artifact plugin by name."""
    return get_artifact_registry().get_plugin(name)


def list_artifact_plugins() -> List[str]:
    """List all available artifact plugins."""
    return get_artifact_registry().list_plugin_names()


__all__ = [
    "ArtifactRegistry",
    "ArtifactPluginInfo",
    "get_artifact_registry",
    "get_artifact_plugin",
    "list_artifact_plugins",
]
