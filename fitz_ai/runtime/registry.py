"""
Engine Registry - Central registry for all knowledge engines.

This module provides a registry pattern for discovering and instantiating
knowledge engines. It enables the platform to support multiple engines
(Classic RAG, CLaRa, custom engines) without hardcoding engine names.

Philosophy:
    - Engines register themselves via decorators
    - Registry provides lookup by name
    - Engines are created lazily (on first use)
    - Configuration is validated at creation time
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from fitz_ai.core import ConfigurationError, KnowledgeEngine


@dataclass
class EngineRegistration:
    """
    Metadata about a registered engine.

    This stores information needed to create an engine instance,
    including the factory function and any metadata.
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


class EngineRegistry:
    """
    Registry for knowledge engines.

    This class maintains a mapping of engine names to factory functions.
    Engines can register themselves using the @register_engine decorator
    or by calling register() directly.

    Examples:
        Register an engine:
        >>> @EngineRegistry.register_engine(
        ...     name="classic_rag",
        ...     description="Retrieval-augmented generation"
        ... )
        ... def create_classic_rag(config):
        ...     return ClassicRagEngine(config)

        Get an engine:
        >>> registry = EngineRegistry.get_global()
        >>> factory = registry.get("classic_rag")
        >>> engine = factory(config)

        List available engines:
        >>> engines = registry.list()
        >>> print(engines)
        ['classic_rag', 'clara', 'custom']
    """

    # Global singleton registry
    _global_registry: Optional["EngineRegistry"] = None

    def __init__(self):
        """Initialize an empty registry."""
        self._engines: Dict[str, EngineRegistration] = {}

    @classmethod
    def get_global(cls) -> "EngineRegistry":
        """
        Get the global singleton registry.

        Returns:
            The global EngineRegistry instance

        Examples:
            >>> registry = EngineRegistry.get_global()
            >>> registry.register(...)
        """
        if cls._global_registry is None:
            cls._global_registry = cls()
        return cls._global_registry

    @classmethod
    def reset_global(cls) -> None:
        """
        Reset the global registry (useful for testing).

        Examples:
            >>> EngineRegistry.reset_global()
            >>> registry = EngineRegistry.get_global()
            >>> assert len(registry.list()) == 0
        """
        cls._global_registry = None

    def register(
        self,
        name: str,
        factory: Callable[[Any], KnowledgeEngine],
        description: str = "",
        config_type: Optional[Type] = None,
    ) -> None:
        """
        Register an engine factory.

        Args:
            name: Unique name for this engine
            factory: Function that creates engine instances (config -> KnowledgeEngine)
            description: Human-readable description
            config_type: Expected config type (optional, for validation)

        Raises:
            ValueError: If an engine with this name is already registered

        Examples:
            >>> def create_my_engine(config):
            ...     return MyEngine(config)
            >>>
            >>> registry = EngineRegistry.get_global()
            >>> registry.register(
            ...     name="my_engine",
            ...     factory=create_my_engine,
            ...     description="My custom engine"
            ... )
        """
        if name in self._engines:
            raise ValueError(f"Engine '{name}' is already registered")

        registration = EngineRegistration(
            name=name, factory=factory, description=description, config_type=config_type
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

        Examples:
            >>> registry = EngineRegistry.get_global()
            >>> factory = registry.get("classic_rag")
            >>> engine = factory(config)
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
            EngineRegistration with full metadata

        Raises:
            ConfigurationError: If no engine with this name is registered

        Examples:
            >>> registry = EngineRegistry.get_global()
            >>> info = registry.get_info("classic_rag")
            >>> print(info.description)
        """
        if name not in self._engines:
            available = ", ".join(self.list())
            raise ConfigurationError(f"Unknown engine: '{name}'. Available engines: {available}")

        return self._engines[name]

    def list(self) -> list[str]:
        """
        List all registered engine names.

        Returns:
            List of engine names

        Examples:
            >>> registry = EngineRegistry.get_global()
            >>> engines = registry.list()
            >>> print(f"Available engines: {', '.join(engines)}")
        """
        return sorted(self._engines.keys())

    def list_with_descriptions(self) -> Dict[str, str]:
        """
        List all engines with their descriptions.

        Returns:
            Dictionary mapping engine names to descriptions

        Examples:
            >>> registry = EngineRegistry.get_global()
            >>> for name, desc in registry.list_with_descriptions().items():
            ...     print(f"{name}: {desc}")
        """
        return {name: reg.description for name, reg in self._engines.items()}

    @staticmethod
    def register_engine(
        name: str,
        description: str = "",
        config_type: Optional[Type] = None,
    ) -> Callable:
        """
        Decorator for registering engine factories.

        This is the recommended way to register engines as it keeps
        registration close to the factory definition.

        Args:
            name: Unique name for this engine
            description: Human-readable description
            config_type: Expected config type (optional)

        Returns:
            Decorator function

        Examples:
            >>> @EngineRegistry.register_engine(
            ...     name="classic_rag",
            ...     description="Retrieval-augmented generation",
            ... )
            ... def create_classic_rag_engine(config):
            ...     return ClassicRagEngine(config)
        """

        def decorator(factory: Callable[[Any], KnowledgeEngine]) -> Callable:
            registry = EngineRegistry.get_global()
            registry.register(
                name=name, factory=factory, description=description, config_type=config_type
            )
            return factory

        return decorator


# Convenience function for global registry
def get_engine_registry() -> EngineRegistry:
    """
    Get the global engine registry.

    This is a convenience function equivalent to EngineRegistry.get_global().

    Returns:
        The global EngineRegistry instance

    Examples:
        >>> registry = get_engine_registry()
        >>> engines = registry.list()
    """
    return EngineRegistry.get_global()
