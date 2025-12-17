"""
Classic RAG Runtime - Canonical entry point for Classic RAG execution.

This module provides the single, stable way to execute Classic RAG queries.
All other entry points (CLI, API, etc.) should route through this runtime.

Philosophy:
    - Single source of truth for RAG execution
    - Clean separation: runtime orchestrates, engine executes
    - All RAG execution flows through run_classic_rag()
"""

from typing import Optional, Dict, Any

from fitz.core import Query, Answer, Constraints

from fitz.engines.classic_rag.config.schema import FitzConfig
from fitz.engines.classic_rag.config.loader import load_config
from fitz.engines.classic_rag.engine import ClassicRagEngine


def run_classic_rag(
        query: str,
        config: Optional[FitzConfig] = None,
        config_path: Optional[str] = None,
        constraints: Optional[Constraints] = None,
        metadata: Optional[Dict[str, Any]] = None,
) -> Answer:
    """
    Execute a Classic RAG query.

    This is the canonical entry point for all Classic RAG execution.
    It handles:
    - Configuration loading (if not provided)
    - Engine initialization
    - Query object construction
    - Answer generation

    Args:
        query: The question text
        config: Optional pre-loaded FitzConfig. If not provided, will load
               from config_path or default location
        config_path: Optional path to config file. Ignored if config provided
        constraints: Optional query-time constraints (filters, limits)
        metadata: Optional engine-specific metadata/hints

    Returns:
        Answer object with generated text and source provenance

    Raises:
        ConfigurationError: If config cannot be loaded
        QueryError: If query is invalid
        KnowledgeError: If retrieval fails
        GenerationError: If answer generation fails

    Examples:
        Simple usage:
        >>> answer = run_classic_rag("What is quantum computing?")
        >>> print(answer.text)

        With custom config:
        >>> config = load_config("my_config.yaml")
        >>> answer = run_classic_rag("Explain entanglement", config=config)

        With constraints:
        >>> constraints = Constraints(max_sources=5, filters={"topic": "physics"})
        >>> answer = run_classic_rag(
        ...     "What is superposition?",
        ...     constraints=constraints
        ... )

        With engine hints:
        >>> answer = run_classic_rag(
        ...     "Summarize the paper",
        ...     metadata={"temperature": 0.3, "model": "claude-3-opus"}
        ... )
    """
    # Load config if not provided
    if config is None:
        config = load_config(config_path)

    # Initialize engine
    engine = ClassicRagEngine(config)

    # Build Query object
    query_obj = Query(
        text=query,
        constraints=constraints,
        metadata=metadata or {}
    )

    # Execute and return
    return engine.answer(query_obj)


def create_classic_rag_engine(
        config: Optional[FitzConfig] = None,
        config_path: Optional[str] = None,
) -> ClassicRagEngine:
    """
    Create and return a Classic RAG engine instance.

    This is useful when you want to:
    - Reuse an engine across multiple queries (more efficient)
    - Access engine internals or configuration
    - Implement custom execution logic

    For simple one-off queries, use run_classic_rag() instead.

    Args:
        config: Optional pre-loaded FitzConfig
        config_path: Optional path to config file

    Returns:
        Initialized ClassicRagEngine instance

    Raises:
        ConfigurationError: If config cannot be loaded or engine initialization fails

    Examples:
        Create and reuse engine:
        >>> engine = create_classic_rag_engine("config.yaml")
        >>>
        >>> # Execute multiple queries with same engine
        >>> q1 = Query(text="What is quantum computing?")
        >>> answer1 = engine.answer(q1)
        >>>
        >>> q2 = Query(text="Explain entanglement")
        >>> answer2 = engine.answer(q2)

        Access engine config:
        >>> engine = create_classic_rag_engine()
        >>> print(engine.config.chat.plugin_name)
    """
    # Load config if not provided
    if config is None:
        config = load_config(config_path)

    # Create and return engine
    return ClassicRagEngine(config)


# Convenience alias for the main entry point
run = run_classic_rag