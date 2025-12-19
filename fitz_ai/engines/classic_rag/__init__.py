"""
Classic RAG Engine - Retrieval-Augmented Generation implementation.

This package implements the classic RAG paradigm as a KnowledgeEngine.

Public API:
    - ClassicRagEngine: The engine implementation
    - run_classic_rag: Simple entry point for one-off queries
    - create_classic_rag_engine: Factory for creating reusable engines

Examples:
    Simple usage:
    >>> from fitz_ai.engines.classic_rag import run_classic_rag
    >>> answer = run_classic_rag("What is quantum computing?")
    >>> print(answer.text)

    Advanced usage:
    >>> from fitz_ai.engines.classic_rag import ClassicRagEngine
    >>> from fitz_ai.engines.classic_rag.config.loader import load_config
    >>> from fitz_ai.core import Query, Constraints
    >>>
    >>> config = load_config("my_config.yaml")
    >>> engine = ClassicRagEngine(config)
    >>>
    >>> query = Query(
    ...     text="Explain entanglement",
    ...     constraints=Constraints(max_sources=5)
    ... )
    >>> answer = engine.answer(query)
"""

from .engine import ClassicRagEngine
from .runtime import create_classic_rag_engine, run, run_classic_rag

__all__ = [
    "ClassicRagEngine",
    "run_classic_rag",
    "create_classic_rag_engine",
    "run",
]
