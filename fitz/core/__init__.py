# fitz/core/__init__.py
"""
Fitz Core - Paradigm-agnostic contracts for knowledge engines.

This module defines the stable abstractions that all engines must implement.
The core philosophy is: Knowledge → Engine → Answer.

Public API:
    - KnowledgeEngine: Protocol that all engines implement
    - Query: Input to engines
    - Answer: Output from engines
    - Provenance: Source attribution
    - Constraints: Query-time constraints
    - FitzPaths: Central path management
    - Exceptions: Standard error hierarchy

Examples:
    Using the core abstractions:
    >>> from fitz.core import Query, Constraints
    >>> from fitz.engines.classic_rag import ClassicRagEngine
    >>>
    >>> # Create an engine (engine-specific)
    >>> engine = ClassicRagEngine(config)
    >>>
    >>> # Use core abstractions (paradigm-agnostic)
    >>> query = Query(
    ...     text="What is quantum computing?",
    ...     constraints=Constraints(max_sources=5)
    ... )
    >>> answer = engine.answer(query)
    >>>
    >>> # Access results (paradigm-agnostic)
    >>> print(answer.text)
    >>> for source in answer.provenance:
    ...     print(f"Source: {source.source_id}")
    
    Using FitzPaths:
    >>> from fitz.core import FitzPaths
    >>> config_path = FitzPaths.config()
    >>> vector_db_path = FitzPaths.vector_db()
"""

# Core protocol
from .engine import KnowledgeEngine

# Core types
from .query import Query
from .answer import Answer
from .provenance import Provenance
from .constraints import Constraints

# Path management
from .paths import FitzPaths, get_workspace, get_vector_db_path, get_config_path

# Core exceptions
from .exceptions import (
    EngineError,
    QueryError,
    KnowledgeError,
    GenerationError,
    ConfigurationError,
    TimeoutError,
    UnsupportedOperationError,
)

__all__ = [
    # Protocol
    "KnowledgeEngine",
    # Types
    "Query",
    "Answer",
    "Provenance",
    "Constraints",
    # Path Management
    "FitzPaths",
    "get_workspace",
    "get_vector_db_path",
    "get_config_path",
    # Exceptions
    "EngineError",
    "QueryError",
    "KnowledgeError",
    "GenerationError",
    "ConfigurationError",
    "TimeoutError",
    "UnsupportedOperationError",
]