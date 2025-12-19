"""
Fitz - Local-First RAG Framework & Engine Platform

Fitz is a paradigm-agnostic knowledge engine platform that supports multiple
approaches to knowledge retrieval and synthesis (Classic RAG, CLaRa, custom engines).

Quick Start:
    >>> from fitz import run
    >>> answer = run("What is quantum computing?")
    >>> print(answer.text)

Public API:
    Core Types:
        - Query: Input to engines
        - Answer: Output from engines
        - Provenance: Source attribution
        - Constraints: Query-time constraints

    Runtime:
        - run: Universal entry point (any engine)
        - create_engine: Factory for creating engines
        - list_engines: List available engines

    Classic RAG:
        - run_classic_rag: RAG-specific entry point
        - create_classic_rag_engine: RAG engine factory

Architecture:
    fitz/
    ├── core/              # Paradigm-agnostic contracts
    ├── engines/           # Engine implementations
    │   └── classic_rag/   # Retrieval-augmented generation
    ├── runtime/           # Multi-engine orchestration
    ├── llm/               # LLM service (chat, embedding, rerank)
    ├── vector_db/         # Vector database service
    └── ingest/            # Document ingestion

Philosophy:
    Knowledge → Engine → Answer

    Engines are black boxes that transform queries into answers.
    The platform only cares about the interface, not the implementation.

Examples:
    Simple query:
    >>> from fitz import run
    >>> answer = run("What is quantum computing?")

    With constraints:
    >>> from fitz import run, Constraints
    >>> constraints = Constraints(max_sources=5)
    >>> answer = run("Explain entanglement", constraints=constraints)

    Specific engine:
    >>> answer = run("What is X?", engine="classic_rag")

    Reusable engine:
    >>> from fitz import create_engine, Query
    >>> engine = create_engine("classic_rag")
    >>> query = Query(text="What is Y?")
    >>> answer = engine.answer(query)
"""

__version__ = "0.3.0"

# =============================================================================
# CORE TYPES
# =============================================================================

from fitz_ai.core import (  # Protocol; Types; Exceptions
    Answer,
    ConfigurationError,
    Constraints,
    EngineError,
    GenerationError,
    KnowledgeEngine,
    KnowledgeError,
    Provenance,
    Query,
    QueryError,
    TimeoutError,
    UnsupportedOperationError,
)
from fitz_ai.engines.classic_rag import (
    ClassicRagEngine,
    create_classic_rag_engine,
    run_classic_rag,
)
from fitz_ai.runtime import (
    create_engine,
    get_engine_registry,
    list_engines,
    list_engines_with_info,
    run,
)

# =============================================================================
# RUNTIME (UNIVERSAL)
# =============================================================================


# =============================================================================
# CLASSIC RAG (CONVENIENCE)
# =============================================================================


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core Protocol
    "KnowledgeEngine",
    # Core Types
    "Query",
    "Answer",
    "Provenance",
    "Constraints",
    # Core Exceptions
    "EngineError",
    "QueryError",
    "KnowledgeError",
    "GenerationError",
    "ConfigurationError",
    "TimeoutError",
    "UnsupportedOperationError",
    # Universal Runtime
    "run",
    "create_engine",
    "list_engines",
    "list_engines_with_info",
    "get_engine_registry",
    # Classic RAG
    "run_classic_rag",
    "create_classic_rag_engine",
    "ClassicRagEngine",
]
