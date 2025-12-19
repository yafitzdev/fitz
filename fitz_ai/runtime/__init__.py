"""
Fitz Runtime - Universal engine orchestration.

This package provides the top-level API for running queries across any engine.
It handles engine discovery, registration, and routing.

Public API:
    - run: Universal entry point for all engines
    - create_engine: Factory for creating engine instances
    - list_engines: Get list of available engines
    - get_engine_registry: Access the global registry

Examples:
    Simple query:
    >>> from fitz_ai.runtime import run
    >>> answer = run("What is quantum computing?")

    Specific engine:
    >>> answer = run("Explain X", engine="clara")

    List engines:
    >>> from fitz_ai.runtime import list_engines
    >>> print(list_engines())
    ['classic_rag', 'clara']

    Create reusable engine:
    >>> from fitz_ai.runtime import create_engine
    >>> engine = create_engine("classic_rag")
    >>> answer = engine.answer(query)
"""

from .registry import (
    EngineRegistration,
    EngineRegistry,
    get_engine_registry,
)
from .runner import (
    create_engine,
    list_engines,
    list_engines_with_info,
    run,
)

__all__ = [
    # Registry
    "EngineRegistry",
    "EngineRegistration",
    "get_engine_registry",
    # Runner
    "run",
    "create_engine",
    "list_engines",
    "list_engines_with_info",
]
