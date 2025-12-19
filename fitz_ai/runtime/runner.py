"""
Universal Runner - Single entry point for all engine execution.

This module provides the top-level API for executing queries across any engine.
It handles engine selection, config loading, and query routing.

Philosophy:
    - One function to rule them all: run()
    - Engine selection is config-driven or explicit
    - All engines accessed through same interface
    - CLI, API, and examples route through here
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from fitz_ai.core import Answer, ConfigurationError, Constraints, Query
from fitz_ai.runtime.registry import get_engine_registry


def run(
    query: Union[str, Query],
    engine: str = "classic_rag",
    config: Optional[Any] = None,
    config_path: Optional[Union[str, Path]] = None,
    constraints: Optional[Constraints] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Answer:
    """
    Universal entry point for executing queries on any engine.

    This function:
    1. Resolves which engine to use
    2. Loads configuration (if needed)
    3. Creates the engine instance
    4. Executes the query
    5. Returns the answer

    Args:
        query: Question text or Query object
        engine: Engine name to use (default: "classic_rag")
        config: Pre-loaded config object (engine-specific type)
        config_path: Path to config file (ignored if config provided)
        constraints: Optional query-time constraints
        metadata: Optional engine-specific hints

    Returns:
        Answer object with text, provenance, and metadata

    Raises:
        ConfigurationError: If engine doesn't exist or config is invalid
        QueryError: If query is invalid
        KnowledgeError: If knowledge retrieval fails
        GenerationError: If answer generation fails

    Examples:
        Simple query:
        >>> answer = run("What is quantum computing?")
        >>> print(answer.text)

        Specific engine:
        >>> answer = run("Explain X", engine="clara")

        With constraints:
        >>> from fitz_ai.core import Constraints
        >>> constraints = Constraints(max_sources=5)
        >>> answer = run("What is Y?", constraints=constraints)

        With config:
        >>> from fitz_ai.engines.classic_rag.config.loader import load_config
        >>> config = load_config("my_config.yaml")
        >>> answer = run("Question?", config=config)
    """
    # Get the registry
    registry = get_engine_registry()

    # Get the factory for the requested engine
    try:
        factory = registry.get(engine)
    except ConfigurationError as e:
        # Provide helpful error with available engines
        available = registry.list()
        raise ConfigurationError(
            f"Unknown engine: '{engine}'. "
            f"Available engines: {', '.join(available)}. "
            f"Original error: {e}"
        ) from e

    # Load config if not provided
    if config is None:
        config = _load_engine_config(engine, config_path)

    # Create engine instance
    engine_instance = factory(config)

    # Build Query object if string was provided
    if isinstance(query, str):
        query_obj = Query(text=query, constraints=constraints, metadata=metadata or {})
    else:
        query_obj = query

    # Execute and return
    return engine_instance.answer(query_obj)


def create_engine(
    engine: str = "classic_rag",
    config: Optional[Any] = None,
    config_path: Optional[Union[str, Path]] = None,
):
    """
    Create an engine instance.

    This is useful when you want to:
    - Reuse an engine across multiple queries (more efficient)
    - Access engine internals or configuration
    - Implement custom execution logic

    For simple one-off queries, use run() instead.

    Args:
        engine: Engine name to create
        config: Pre-loaded config object
        config_path: Path to config file

    Returns:
        Engine instance implementing KnowledgeEngine protocol

    Raises:
        ConfigurationError: If engine doesn't exist or config is invalid

    Examples:
        Create and reuse engine:
        >>> engine = create_engine("classic_rag", config_path="config.yaml")
        >>>
        >>> q1 = Query(text="What is quantum computing?")
        >>> answer1 = engine.answer(q1)
        >>>
        >>> q2 = Query(text="Explain entanglement")
        >>> answer2 = engine.answer(q2)

        Create specific engine:
        >>> clara_engine = create_engine("clara")
        >>> answer = clara_engine.answer(query)
    """
    # Get the registry
    registry = get_engine_registry()

    # Get the factory
    factory = registry.get(engine)

    # Load config if not provided
    if config is None:
        config = _load_engine_config(engine, config_path)

    # Create and return
    return factory(config)


def list_engines() -> list[str]:
    """
    List all available engines.

    Returns:
        List of engine names

    Examples:
        >>> engines = list_engines()
        >>> print(f"Available: {', '.join(engines)}")
        ['classic_rag', 'clara', 'custom']
    """
    registry = get_engine_registry()
    return registry.list()


def list_engines_with_info() -> Dict[str, str]:
    """
    List all engines with descriptions.

    Returns:
        Dictionary mapping engine names to descriptions

    Examples:
        >>> for name, desc in list_engines_with_info().items():
        ...     print(f"{name}: {desc}")
        classic_rag: Retrieval-augmented generation
        clara: Citation-attributed reasoning
    """
    registry = get_engine_registry()
    return registry.list_with_descriptions()


def _load_engine_config(engine_name: str, config_path: Optional[Union[str, Path]] = None):
    """
    Load configuration for a specific engine.

    This is an internal helper that routes to engine-specific config loaders.
    For engines that don't need config or for testing, returns None.

    Args:
        engine_name: Name of the engine
        config_path: Optional path to config file

    Returns:
        Loaded config object (engine-specific type) or None

    Raises:
        ConfigurationError: If config cannot be loaded for known engines
    """
    # Import here to avoid circular dependencies
    if engine_name == "classic_rag":
        from fitz_ai.engines.classic_rag.config.loader import load_config as load_rag_config

        return load_rag_config(str(config_path) if config_path else None)

    elif engine_name == "clara":
        # Future: CLaRa config loader
        # from fitz_ai.engines.clara.config.loader import load_config as load_clara_config
        # return load_clara_config(str(config_path) if config_path else None)
        raise ConfigurationError(f"Config loading not yet implemented for engine: {engine_name}")

    else:
        # For custom engines, return None - they should provide config explicitly
        # This allows testing with mock engines that don't need config
        return None
