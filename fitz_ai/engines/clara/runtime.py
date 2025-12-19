# fitz/engines/clara/runtime.py
"""
CLaRa Engine Runtime - Convenience functions for CLaRa engine.

This module provides simple entry points for using CLaRa, following
the same pattern as classic_rag/runtime.py.
"""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from fitz_ai.core import Answer, Constraints, Query

# Import directly from schema to avoid circular imports through __init__
from fitz_ai.engines.clara.config.schema import ClaraConfig, load_clara_config

if TYPE_CHECKING:
    from fitz_ai.engines.clara.engine import ClaraEngine

# Module-level engine cache for reuse
_cached_engine: Optional["ClaraEngine"] = None
_cached_config_path: Optional[str] = None


def run_clara(
    query: Union[str, Query],
    documents: Optional[List[str]] = None,
    config: Optional[ClaraConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    constraints: Optional[Constraints] = None,
) -> Answer:
    """
    Run a query through the CLaRa engine.

    This is the simplest way to use CLaRa. For repeated queries,
    use create_clara_engine() to avoid re-loading the model.

    Args:
        query: Question text or Query object
        documents: List of documents to search. Required on first call
                   unless using a pre-loaded knowledge base.
        config: Pre-built ClaraConfig object
        config_path: Path to YAML config file
        constraints: Optional query-time constraints

    Returns:
        Answer object with text and provenance

    Examples:
        Basic usage:
        >>> answer = run_clara(
        ...     "What is quantum computing?",
        ...     documents=["Quantum computing uses qubits...", "..."]
        ... )
        >>> print(answer.text)

        With config:
        >>> config = ClaraConfig(
        ...     compression=ClaraCompressionConfig(compression_rate=32)
        ... )
        >>> answer = run_clara("Question?", documents=docs, config=config)
    """
    global _cached_engine, _cached_config_path

    # Import here to avoid circular imports
    from fitz_ai.engines.clara.engine import ClaraEngine

    # Determine config path for caching
    current_config_path = str(config_path) if config_path else None

    # Reuse cached engine if config hasn't changed
    if _cached_engine is not None and _cached_config_path == current_config_path and config is None:
        engine = _cached_engine
    else:
        # Create new engine
        if config is None:
            config = load_clara_config(current_config_path)

        engine = ClaraEngine(config)
        _cached_engine = engine
        _cached_config_path = current_config_path

    # Add documents if provided
    if documents:
        engine.add_documents(documents)

    # Build query
    if isinstance(query, str):
        query_obj = Query(text=query, constraints=constraints)
    else:
        query_obj = query
        if constraints:
            query_obj = Query(
                text=query_obj.text,
                constraints=constraints,
                metadata=query_obj.metadata,
            )

    return engine.answer(query_obj)


def create_clara_engine(
    config: Optional[ClaraConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> "ClaraEngine":
    """
    Create a CLaRa engine instance.

    Use this when you want to:
    - Reuse the engine across multiple queries (recommended)
    - Pre-load documents into the knowledge base
    - Access engine internals

    Args:
        config: Pre-built ClaraConfig object
        config_path: Path to YAML config file

    Returns:
        ClaraEngine instance

    Examples:
        Create and use engine:
        >>> engine = create_clara_engine()
        >>> engine.add_documents(my_documents)
        >>>
        >>> answer1 = engine.answer(Query(text="Question 1?"))
        >>> answer2 = engine.answer(Query(text="Question 2?"))

        With custom config:
        >>> config = ClaraConfig(
        ...     model=ClaraModelConfig(model_name_or_path="apple/CLaRa-7B-Instruct")
        ... )
        >>> engine = create_clara_engine(config=config)
    """
    # Import here to avoid circular imports
    from fitz_ai.engines.clara.engine import ClaraEngine

    if config is None:
        config = load_clara_config(str(config_path) if config_path else None)

    return ClaraEngine(config)


def clear_engine_cache() -> None:
    """
    Clear the cached engine.

    Call this to free memory or force re-initialization on next run.
    """
    global _cached_engine, _cached_config_path
    _cached_engine = None
    _cached_config_path = None
