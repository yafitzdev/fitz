# fitz_ai/runtime/registry.py
"""
Engine Registry - Central registry for all knowledge engines.

This module provides a registry pattern for discovering and instantiating
knowledge engines. It enables the platform to support multiple engines
(Classic RAG, CLaRa, custom engines) without hardcoding engine names.

Philosophy:
    - Engines register themselves on import
    - Registry provides lookup by name
    - Engines are created lazily (on first use)
    - Engines declare their capabilities for CLI/API adaptation
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from fitz_ai.core import ConfigurationError, KnowledgeEngine


@dataclass
class EngineCapabilities:
    """
    Capabilities declared by an engine.

    These flags tell the CLI and API how to interact with the engine,
    enabling generic command handling without engine-specific if/elif chains.
    """

    # Storage capabilities
    supports_collections: bool = True
    """Engine uses persistent collections (vector DB, etc.)"""

    requires_documents_at_query: bool = False
    """Engine needs documents added before querying (no persistent storage)"""

    # Feature capabilities
    supports_chat: bool = True
    """Engine supports multi-turn conversation"""

    supports_streaming: bool = False
    """Engine supports streaming responses"""

    # Requirements
    requires_config: bool = True
    """Engine requires a config file to be present"""

    requires_api_key: bool = False
    """Engine requires an API key (e.g., COHERE_API_KEY)"""

    api_key_env_var: Optional[str] = None
    """Environment variable name for API key"""

    # CLI hints
    cli_query_message: Optional[str] = None
    """Custom message to show in query command (e.g., 'use quickstart instead')"""


@dataclass
class EngineRegistration:
    """
    Metadata about a registered engine.

    This stores everything needed to create and interact with an engine,
    including factory function, config loader, and capabilities.
    """

    name: str
    """Unique name for this engine (e.g., 'classic_rag', 'clara')."""

    factory: Callable[[Any], KnowledgeEngine]
    """
    Factory function that creates an engine instance.

    Signature: (config: Any) -> KnowledgeEngine
    The config type is engine-specific.
    """

    description: str
    """Human-readable description of what this engine does."""

    config_type: Optional[Type] = None
    """Expected config type (for validation). Optional."""

    config_loader: Optional[Callable[[Optional[str]], Any]] = None
    """
    Function to load config for this engine.

    Signature: (config_path: Optional[str]) -> config
    If None, engine doesn't need config loading.
    """

    capabilities: EngineCapabilities = field(default_factory=EngineCapabilities)
    """Engine capabilities for CLI/API adaptation."""


class EngineRegistry:
    """
    Registry for knowledge engines.

    This class maintains a mapping of engine names to factory functions.
    Engines can register themselves using the @register_engine decorator
    or by calling register() directly.

    Examples:
        Register an engine:
        >>> @EngineRegistry.register_engine(
        ...     name="my_engine",
        ...     description="My custom engine",
        ...     capabilities=EngineCapabilities(supports_collections=False),
        ... )
        ... def create_my_engine(config):
        ...     return MyEngine(config)

        Get an engine:
        >>> registry = EngineRegistry.get_global()
        >>> factory = registry.get("my_engine")
        >>> engine = factory(config)

        Check capabilities:
        >>> info = registry.get_info("my_engine")
        >>> if info.capabilities.supports_collections:
        ...     # Show collection picker
    """

    # Global singleton registry
    _global_registry: Optional["EngineRegistry"] = None

    def __init__(self):
        """Initialize an empty registry."""
        self._engines: Dict[str, EngineRegistration] = {}

    @classmethod
    def get_global(cls) -> "EngineRegistry":
        """Get the global singleton registry."""
        if cls._global_registry is None:
            cls._global_registry = cls()
        return cls._global_registry

    @classmethod
    def reset_global(cls) -> None:
        """Reset the global registry (useful for testing)."""
        cls._global_registry = None

    def register(
        self,
        name: str,
        factory: Callable[[Any], KnowledgeEngine],
        description: str = "",
        config_type: Optional[Type] = None,
        config_loader: Optional[Callable[[Optional[str]], Any]] = None,
        capabilities: Optional[EngineCapabilities] = None,
    ) -> None:
        """
        Register an engine factory.

        Args:
            name: Unique name for this engine
            factory: Function that creates engine instances (config -> KnowledgeEngine)
            description: Human-readable description
            config_type: Expected config type (optional, for validation)
            config_loader: Function to load config (config_path -> config)
            capabilities: Engine capabilities for CLI/API adaptation

        Raises:
            ValueError: If an engine with this name is already registered
        """
        if name in self._engines:
            raise ValueError(f"Engine '{name}' is already registered")

        registration = EngineRegistration(
            name=name,
            factory=factory,
            description=description,
            config_type=config_type,
            config_loader=config_loader,
            capabilities=capabilities or EngineCapabilities(),
        )
        self._engines[name] = registration

    def get(self, name: str) -> Callable[[Any], KnowledgeEngine]:
        """
        Get the factory function for an engine.

        Args:
            name: Name of the engine to retrieve

        Returns:
            Factory function that creates engine instances

        Raises:
            ConfigurationError: If no engine with this name is registered
        """
        if name not in self._engines:
            available = ", ".join(self.list())
            raise ConfigurationError(f"Unknown engine: '{name}'. Available engines: {available}")

        return self._engines[name].factory

    def get_info(self, name: str) -> EngineRegistration:
        """
        Get full registration info for an engine.

        Args:
            name: Name of the engine

        Returns:
            EngineRegistration with full metadata including capabilities

        Raises:
            ConfigurationError: If no engine with this name is registered
        """
        if name not in self._engines:
            available = ", ".join(self.list())
            raise ConfigurationError(f"Unknown engine: '{name}'. Available engines: {available}")

        return self._engines[name]

    def get_capabilities(self, name: str) -> EngineCapabilities:
        """
        Get capabilities for an engine.

        Args:
            name: Name of the engine

        Returns:
            EngineCapabilities object

        Raises:
            ConfigurationError: If no engine with this name is registered
        """
        return self.get_info(name).capabilities

    def load_config(self, name: str, config_path: Optional[str] = None) -> Any:
        """
        Load configuration for an engine using its registered loader.

        Args:
            name: Name of the engine
            config_path: Optional path to config file

        Returns:
            Loaded config object (engine-specific type)

        Raises:
            ConfigurationError: If engine not found or has no config loader
        """
        info = self.get_info(name)

        if info.config_loader is None:
            # No config loader - return None (engine will use defaults)
            return None

        return info.config_loader(config_path)

    def list(self) -> List[str]:
        """List all registered engine names."""
        return sorted(self._engines.keys())

    def list_with_descriptions(self) -> Dict[str, str]:
        """List all engines with their descriptions."""
        return {name: reg.description for name, reg in self._engines.items()}

    def list_with_capabilities(self) -> Dict[str, EngineCapabilities]:
        """List all engines with their capabilities."""
        return {name: reg.capabilities for name, reg in self._engines.items()}

    @staticmethod
    def register_engine(
        name: str,
        description: str = "",
        config_type: Optional[Type] = None,
        config_loader: Optional[Callable[[Optional[str]], Any]] = None,
        capabilities: Optional[EngineCapabilities] = None,
    ) -> Callable:
        """
        Decorator for registering engine factories.

        This is the recommended way to register engines as it keeps
        registration close to the factory definition.

        Args:
            name: Unique name for this engine
            description: Human-readable description
            config_type: Expected config type (optional)
            config_loader: Function to load config
            capabilities: Engine capabilities

        Returns:
            Decorator function

        Examples:
            >>> @EngineRegistry.register_engine(
            ...     name="my_engine",
            ...     description="My custom RAG engine",
            ...     capabilities=EngineCapabilities(
            ...         supports_collections=False,
            ...         requires_documents_at_query=True,
            ...     ),
            ... )
            ... def create_my_engine(config):
            ...     return MyEngine(config)
        """

        def decorator(factory: Callable[[Any], KnowledgeEngine]) -> Callable:
            registry = EngineRegistry.get_global()
            registry.register(
                name=name,
                factory=factory,
                description=description,
                config_type=config_type,
                config_loader=config_loader,
                capabilities=capabilities,
            )
            return factory

        return decorator


# Convenience function for global registry
def get_engine_registry() -> EngineRegistry:
    """Get the global engine registry."""
    return EngineRegistry.get_global()
