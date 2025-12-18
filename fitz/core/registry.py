# fitz/core/registry.py
"""
Centralized Plugin Registry System.

Single implementation of plugin registry logic used by all plugin types:
- Ingest
- Chunking
- Retriever
- Pipeline
- LLM (YAML-based, via fitz.llm.registry)
- Vector DB (YAML-based, via fitz.vector_db.registry)

This module is the SINGLE SOURCE OF TRUTH for all registry access.
Import everything from here, not from domain-specific modules.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class PluginRegistryError(Exception):
    """Base error for plugin registry operations."""

    pass


class PluginNotFoundError(PluginRegistryError):
    """Raised when requested plugin doesn't exist."""

    pass


class DuplicatePluginError(PluginRegistryError):
    """Raised when two plugins have the same name."""

    pass


class LLMRegistryError(PluginNotFoundError):
    """Error from LLM registry."""

    pass


class VectorDBRegistryError(PluginNotFoundError):
    """Error from Vector DB registry."""

    pass


# =============================================================================
# PluginRegistry Class
# =============================================================================


@dataclass
class PluginRegistry:
    """
    Generic plugin registry with lazy auto-discovery.

    Args:
        name: Registry name (for error messages)
        scan_packages: List of package names to scan for plugins
        required_method: Method name that plugins must have
        plugin_name_attr: Attribute containing the plugin name
        plugin_type_filter: If set, only register plugins with this plugin_type
        check_module_match: If True, only register classes defined in the scanned module
    """

    name: str
    scan_packages: List[str]
    required_method: str
    plugin_name_attr: str = "plugin_name"
    plugin_type_filter: str | None = None
    check_module_match: bool = True
    _plugins: Dict[str, Type[Any]] = field(default_factory=dict, repr=False)
    _discovered: bool = field(default=False, repr=False)

    def get(self, plugin_name: str) -> Type[Any]:
        """Get a plugin by name."""
        self._ensure_discovered()

        if plugin_name not in self._plugins:
            available = sorted(self._plugins.keys())
            raise PluginNotFoundError(
                f"Unknown {self.name} plugin: {plugin_name!r}. Available: {available}"
            )

        return self._plugins[plugin_name]

    def list_available(self) -> List[str]:
        """List all available plugin names."""
        self._ensure_discovered()
        return sorted(self._plugins.keys())

    def register(self, plugin_class: Type[Any]) -> None:
        """Manually register a plugin class."""
        if not hasattr(plugin_class, self.required_method):
            raise PluginRegistryError(
                f"{self.name} plugin {plugin_class.__name__} missing required "
                f"method {self.required_method!r}"
            )

        if not hasattr(plugin_class, self.plugin_name_attr):
            raise PluginRegistryError(
                f"{self.name} plugin {plugin_class.__name__} missing required "
                f"attribute {self.plugin_name_attr!r}"
            )

        name = getattr(plugin_class, self.plugin_name_attr)

        if name in self._plugins:
            existing = self._plugins[name]
            if existing is not plugin_class:
                raise DuplicatePluginError(
                    f"Duplicate {self.name} plugin: {name!r}. "
                    f"Found in {existing.__module__} and {plugin_class.__module__}"
                )
            return

        self._plugins[name] = plugin_class
        logger.debug(f"Registered {self.name} plugin: {name!r}")

    def _ensure_discovered(self) -> None:
        """Run auto-discovery if not already done."""
        if self._discovered:
            return

        for package_name in self.scan_packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.debug(f"Could not import {package_name}: {e}")
                continue

            self._scan_package(package)

        self._discovered = True
        logger.debug(
            f"Discovered {len(self._plugins)} {self.name} plugin(s): "
            f"{sorted(self._plugins.keys())}"
        )

    def _scan_package(self, package: Any) -> None:
        """Scan a package for plugin classes."""
        package_path = getattr(package, "__path__", None)
        if not package_path:
            return

        for importer, modname, ispkg in pkgutil.walk_packages(
            package_path, prefix=f"{package.__name__}."
        ):
            if ispkg:
                continue

            try:
                module = importlib.import_module(modname)
            except Exception as e:
                logger.debug(f"Could not import {modname}: {e}")
                continue

            self._scan_module(module)

    def _scan_module(self, module: Any) -> None:
        """Scan a module for plugin classes."""
        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)

            if not isinstance(obj, type):
                continue

            if not hasattr(obj, self.required_method):
                continue

            if not hasattr(obj, self.plugin_name_attr):
                continue

            if self.plugin_type_filter:
                plugin_type = getattr(obj, "plugin_type", None)
                if plugin_type != self.plugin_type_filter:
                    continue

            if self.check_module_match:
                if obj.__module__ != module.__name__:
                    continue

            try:
                self.register(obj)
            except (PluginRegistryError, DuplicatePluginError) as e:
                logger.debug(f"Skipping {name}: {e}")


# =============================================================================
# Pre-configured Registries (Python-based plugins)
# =============================================================================


INGEST_REGISTRY = PluginRegistry(
    name="ingest",
    scan_packages=["fitz.ingest.ingestion.plugins"],
    required_method="ingest",
)

CHUNKING_REGISTRY = PluginRegistry(
    name="chunking",
    scan_packages=["fitz.ingest.chunking.plugins"],
    required_method="chunk_text",
)

RETRIEVER_REGISTRY = PluginRegistry(
    name="retriever",
    scan_packages=["fitz.engines.classic_rag.retrieval.runtime.plugins"],
    required_method="retrieve",
)

PIPELINE_REGISTRY = PluginRegistry(
    name="pipeline",
    scan_packages=["fitz.engines.classic_rag.pipeline.pipeline.plugins"],
    required_method="build",
)


# =============================================================================
# Convenience Functions for Python-based Registries
# =============================================================================


def get_ingest_plugin(plugin_name: str) -> Type[Any]:
    """Get an ingestion plugin by name."""
    return INGEST_REGISTRY.get(plugin_name)


def available_ingest_plugins() -> List[str]:
    """List available ingestion plugins."""
    return INGEST_REGISTRY.list_available()


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunking plugin by name."""
    return CHUNKING_REGISTRY.get(plugin_name)


def available_chunking_plugins() -> List[str]:
    """List available chunking plugins."""
    return CHUNKING_REGISTRY.list_available()


def get_retriever_plugin(plugin_name: str) -> Type[Any]:
    """Get a retriever plugin by name."""
    return RETRIEVER_REGISTRY.get(plugin_name)


def available_retriever_plugins() -> List[str]:
    """List available retriever plugins."""
    return RETRIEVER_REGISTRY.list_available()


def get_pipeline_plugin(plugin_name: str) -> Type[Any]:
    """Get a pipeline plugin by name."""
    return PIPELINE_REGISTRY.get(plugin_name)


def available_pipeline_plugins() -> List[str]:
    """List available pipeline plugins."""
    return PIPELINE_REGISTRY.list_available()


# =============================================================================
# Re-exports from YAML-based Registries (LLM and Vector DB)
# =============================================================================
# These are lazy imports to avoid circular dependencies at module load time.


def get_llm_plugin(plugin_name: str, plugin_type: str, **kwargs: Any) -> Any:
    """
    Get an LLM plugin instance.

    Args:
        plugin_name: Name of the plugin (e.g., 'openai', 'anthropic', 'local')
        plugin_type: Type of plugin ('chat', 'embedding', 'rerank')
        **kwargs: Plugin configuration

    Returns:
        LLM plugin instance
    """
    from fitz.llm.registry import get_llm_plugin as _get_llm_plugin

    return _get_llm_plugin(plugin_name=plugin_name, plugin_type=plugin_type, **kwargs)


def available_llm_plugins(plugin_type: str) -> List[str]:
    """
    List available LLM plugins for a given type.

    Args:
        plugin_type: Type of plugin ('chat', 'embedding', 'rerank')

    Returns:
        Sorted list of plugin names
    """
    from fitz.llm.registry import available_llm_plugins as _available_llm_plugins

    return _available_llm_plugins(plugin_type)


def get_vector_db_plugin(plugin_name: str, **kwargs: Any) -> Any:
    """
    Get a vector DB plugin instance.

    Args:
        plugin_name: Name of the plugin (e.g., 'qdrant', 'pinecone', 'local-faiss')
        **kwargs: Plugin configuration (host, port, etc.)

    Returns:
        Vector DB plugin instance
    """
    from fitz.vector_db.registry import get_vector_db_plugin as _get_vector_db_plugin

    return _get_vector_db_plugin(plugin_name, **kwargs)


def available_vector_db_plugins() -> List[str]:
    """
    List available vector DB plugins.

    Returns:
        Sorted list of plugin names
    """
    from fitz.vector_db.registry import available_vector_db_plugins as _available_vector_db_plugins

    return _available_vector_db_plugins()


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Exceptions
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
    "LLMRegistryError",
    "VectorDBRegistryError",
    # Registry class
    "PluginRegistry",
    # Pre-configured registries
    "INGEST_REGISTRY",
    "CHUNKING_REGISTRY",
    "RETRIEVER_REGISTRY",
    "PIPELINE_REGISTRY",
    # Python-based plugin accessors
    "get_ingest_plugin",
    "available_ingest_plugins",
    "get_chunking_plugin",
    "available_chunking_plugins",
    "get_retriever_plugin",
    "available_retriever_plugins",
    "get_pipeline_plugin",
    "available_pipeline_plugins",
    # YAML-based plugin accessors (re-exported)
    "get_llm_plugin",
    "available_llm_plugins",
    "get_vector_db_plugin",
    "available_vector_db_plugins",
]
