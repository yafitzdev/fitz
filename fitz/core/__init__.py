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
    - Utils: extract_path, set_nested_path

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

    Using extract_path:
    >>> from fitz.core import extract_path
    >>> data = {"response": {"choices": [{"text": "Hello"}]}}
    >>> extract_path(data, "response.choices[0].text")
    'Hello'
"""

from .answer import Answer
from .constraints import Constraints

# Core protocol
from .engine import KnowledgeEngine

# Core exceptions
from .exceptions import (
    ConfigurationError,
    EngineError,
    GenerationError,
    KnowledgeError,
    QueryError,
    TimeoutError,
    UnsupportedOperationError,
)

# Path management
from .paths import FitzPaths, get_config_path, get_vector_db_path, get_workspace
from .provenance import Provenance

# Core types
from .query import Query

# Core utilities
from .utils import extract_path, set_nested_path

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
    # Utilities
    "extract_path",
    "set_nested_path",
]
