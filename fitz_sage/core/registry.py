# fitz_sage/core/registry.py
"""
Centralized Plugin Registry System.

Single implementation of plugin registry logic used by all plugin types:
- Ingest
- Chunking
- Retriever
- Pipeline
- LLM (Python providers, via fitz_sage.llm)
- Vector DB (YAML-based, via fitz_sage.vector_db.registry)

This module is the SINGLE SOURCE OF TRUTH for all registry access.
Import everything from here, not from domain-specific modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

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
        user_plugin_path: Optional path to user plugins directory (scanned as files)
    """

    name: str
    scan_packages: List[str]
    required_method: str
    plugin_name_attr: str = "plugin_name"
    plugin_type_filter: str | None = None
    check_module_match: bool = True
    user_plugin_paths: List[Path] = field(default_factory=list)
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

        # Scan package plugins first
        for package_name in self.scan_packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.debug(f"Could not import {package_name}: {e}")
                continue

            self._scan_package(package)

        # Scan user plugins (higher priority - can override package plugins)
        for plugin_path in self.user_plugin_paths:
            if plugin_path.exists():
                self._scan_user_plugins(plugin_path)

        self._discovered = True
        logger.debug(
            f"Discovered {len(self._plugins)} {self.name} plugin(s): {sorted(self._plugins.keys())}"
        )

    def _scan_user_plugins(self, plugins_dir: Path) -> None:
        """Scan a directory for user plugin files."""
        for py_file in plugins_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            module_name = f"fitz_user_plugins.{self.name}.{py_file.stem}"

            try:
                module = self._load_module_from_file(py_file, module_name)
            except Exception as e:
                logger.debug(f"Could not load user plugin {py_file}: {e}")
                continue

            # Scan with check_module_match=False since these are dynamically loaded
            self._scan_module(module, check_module_match=False)

    def _load_module_from_file(self, file_path: Path, module_name: str) -> Any:
        """Load a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module

    def _scan_package(self, package: Any) -> None:
        """Scan a package for plugin classes (non-recursive)."""
        package_path = getattr(package, "__path__", None)
        if not package_path:
            return

        # Use iter_modules instead of walk_packages to avoid recursing into subpackages
        for importer, modname, ispkg in pkgutil.iter_modules(
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

    def _scan_module(self, module: Any, check_module_match: Optional[bool] = None) -> None:
        """
        Scan a module for plugin classes.

        Args:
            module: The module to scan
            check_module_match: Override for self.check_module_match (for user plugins)
        """
        should_check_module = (
            check_module_match if check_module_match is not None else self.check_module_match
        )

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

            if should_check_module:
                if obj.__module__ != module.__name__:
                    continue

            try:
                self.register(obj)
            except (PluginRegistryError, DuplicatePluginError) as e:
                logger.debug(f"Skipping {name}: {e}")


# =============================================================================
# Pre-configured Registries (Python-based plugins)
# =============================================================================


def _get_plugin_paths(plugin_type: str) -> list[Path]:
    """Get plugin paths for a given plugin type (~/.fitz/plugins/)."""
    from fitz_sage.core.paths import FitzPaths

    paths = []

    # User home plugins (~/.fitz/plugins/) - same for everyone
    user_map = {
        "chunking": FitzPaths.user_chunking_plugins,
        "constraint": FitzPaths.user_constraint_plugins,
    }
    getter = user_map.get(plugin_type)
    if getter:
        user_path = getter()
        if user_path.exists():
            paths.append(user_path)

    return paths


# Default chunkers (simple, recursive) - shown in fitz init
CHUNKING_REGISTRY = PluginRegistry(
    name="chunking",
    scan_packages=["fitz_sage.ingestion.chunking.plugins.default"],
    required_method="chunk",
    user_plugin_paths=_get_plugin_paths("chunking"),
)

# Type-specific chunkers (markdown, python_code, pdf_sections)
# These are in the top-level plugins/ folder but NOT shown in fitz init.
# Used via by_extension config for file-type-specific chunking.
TYPED_CHUNKING_REGISTRY = PluginRegistry(
    name="typed_chunking",
    scan_packages=["fitz_sage.ingestion.chunking.plugins"],
    required_method="chunk",
    user_plugin_paths=_get_plugin_paths("chunking"),
)

RETRIEVER_REGISTRY = PluginRegistry(
    name="retriever",
    scan_packages=["fitz_sage.engines.fitz_krag.retrieval.strategies"],
    required_method="retrieve",
)

PIPELINE_REGISTRY = PluginRegistry(
    name="pipeline",
    scan_packages=["fitz_sage.engines.fitz_krag"],
    required_method="build",
)

# =============================================================================
# Convenience Functions for Python-based Registries
# =============================================================================


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunking plugin by name (default or typed)."""
    # Try default registry first
    try:
        return CHUNKING_REGISTRY.get(plugin_name)
    except PluginNotFoundError:
        pass
    # Fall back to typed registry
    return TYPED_CHUNKING_REGISTRY.get(plugin_name)


def available_chunking_plugins() -> List[str]:
    """List available default chunking plugins (for fitz init)."""
    return CHUNKING_REGISTRY.list_available()


def get_typed_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a typed chunking plugin by name."""
    return TYPED_CHUNKING_REGISTRY.get(plugin_name)


def available_typed_chunking_plugins() -> List[str]:
    """List available typed chunking plugins (for by_extension config)."""
    return TYPED_CHUNKING_REGISTRY.list_available()


def get_retriever_plugin(plugin_name: str) -> Type[Any]:
    """Get a retriever plugin by name."""
    return RETRIEVER_REGISTRY.get(plugin_name)


def available_retrieval_plugins() -> List[str]:
    """List available retriever plugins."""
    return RETRIEVER_REGISTRY.list_available()


def get_pipeline_plugin(plugin_name: str) -> Type[Any]:
    """Get a pipeline plugin by name."""
    return PIPELINE_REGISTRY.get(plugin_name)


def available_pipeline_plugins() -> List[str]:
    """List available pipeline plugins."""
    return PIPELINE_REGISTRY.list_available()


# =============================================================================
# LLM Plugin Discovery
# =============================================================================

# Provider capabilities mapping
_LLM_PROVIDER_CAPABILITIES: Dict[str, List[str]] = {
    "endpoint": ["chat", "vision"],
    "openai": ["chat", "vision"],
    "azure_openai": ["chat", "vision"],
    "enterprise": ["chat", "vision"],
}


def available_llm_plugins(plugin_type: str) -> List[str]:
    """
    List available LLM plugins for a given type.

    Args:
        plugin_type: Type of LLM plugin ("chat", "embedding", "rerank", "vision")

    Returns:
        List of provider names that support the given type
    """
    return [
        provider
        for provider, capabilities in _LLM_PROVIDER_CAPABILITIES.items()
        if plugin_type in capabilities
    ]


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    # Exceptions
    "PluginRegistryError",
    "PluginNotFoundError",
    "DuplicatePluginError",
    # Registry class
    "PluginRegistry",
    # Pre-configured registries
    "CHUNKING_REGISTRY",
    "TYPED_CHUNKING_REGISTRY",
    "RETRIEVER_REGISTRY",
    "PIPELINE_REGISTRY",
    # Python-based plugin accessors
    "get_chunking_plugin",
    "available_chunking_plugins",
    "get_typed_chunking_plugin",
    "available_typed_chunking_plugins",
    "get_retriever_plugin",
    "available_retrieval_plugins",
    "get_pipeline_plugin",
    "available_pipeline_plugins",
    # LLM plugin accessors
    "available_llm_plugins",
]
