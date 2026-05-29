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

from fitz_sage.core import Answer, ConfigurationError, Query
from fitz_sage.runtime.registry import get_engine_registry


def run(
    query: Union[str, Query],
    engine: str | None = None,
    config: Optional[Any] = None,
    config_path: Optional[Union[str, Path]] = None,
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
        engine: Engine name to use (None = user's default engine)
        config: Pre-loaded config object (engine-specific type)
        config_path: Path to config file (ignored if config provided)
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
        >>> answer = run("Explain X", engine="custom")

        With config:
        >>> from fitz_sage.config.loader import load_engine_config
        >>> config = load_engine_config("fitz_krag")
        >>> answer = run("Question?", config=config)
    """
    # Resolve engine name
    if engine is None:
        from fitz_sage.runtime.registry import get_default_engine

        engine = get_default_engine()

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
        query_obj = Query(text=query, metadata=metadata or {})
    else:
        query_obj = query

    # Execute and return
    return engine_instance.answer(query_obj)


def create_engine(
    engine: str | None = None,
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
        engine: Engine name to create (None = user's default engine)
        config: Pre-loaded config object
        config_path: Path to config file

    Returns:
        Engine instance implementing KnowledgeEngine protocol

    Raises:
        ConfigurationError: If engine doesn't exist or config is invalid

    Examples:
        Create and reuse engine:
        >>> engine = create_engine("fitz_krag", config_path="config.yaml")
        >>>
        >>> q1 = Query(text="What is quantum computing?")
        >>> answer1 = engine.answer(q1)
        >>>
        >>> q2 = Query(text="Explain entanglement")
        >>> answer2 = engine.answer(q2)

        Create specific engine:
        >>> custom_engine = create_engine("custom")
        >>> answer = custom_engine.answer(query)
    """
    # Resolve engine name
    if engine is None:
        from fitz_sage.runtime.registry import get_default_engine

        engine = get_default_engine()

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
        ['fitz_krag']
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
        fitz_krag: Knowledge routing augmented generation
    """
    registry = get_engine_registry()
    return registry.list_with_descriptions()


def _load_engine_config(engine_name: str, config_path: Optional[Union[str, Path]] = None):
    """
    Load configuration for a specific engine.

    This uses the engine's registered config_loader function.
    No hardcoding of engine names - fully extensible.

    Args:
        engine_name: Name of the engine
        config_path: Optional path to config file

    Returns:
        Loaded config object (engine-specific type) or None

    Raises:
        ConfigurationError: If engine not found
    """
    registry = get_engine_registry()
    return registry.load_config(engine_name, str(config_path) if config_path else None)
