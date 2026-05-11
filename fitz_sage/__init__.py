# fitz_sage/__init__.py
"""
Fitz - Local-First RAG Framework & Engine Platform

Fitz is a paradigm-agnostic knowledge engine platform that supports multiple
approaches to knowledge retrieval and synthesis.

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

Architecture:
    fitz_sage/
    ├── core/              # Paradigm-agnostic contracts
    ├── engines/           # Engine implementations
    │   └── fitz_krag/     # Knowledge Routing Augmented Generation
    ├── runtime/           # Multi-engine orchestration
    ├── llm/               # LLM service (chat, rerank, vision)
    ├── storage/           # SQLite connection management
    └── ingestion/         # Document ingestion

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
    >>> answer = run("What is X?", engine="fitz_krag")

    Reusable engine:
    >>> from fitz import create_engine, Query
    >>> engine = create_engine("fitz_krag")
    >>> query = Query(text="What is Y?")
    >>> answer = engine.answer(query)
"""

__version__ = "0.11.0"

# =============================================================================
# LAZY IMPORTS
# =============================================================================
# Heavy modules (engines, runtime) are only imported when accessed.
# This keeps CLI startup fast.


def __getattr__(name: str):
    """Lazy import for heavy modules."""
    # Core types (lightweight, always available)
    if name in (
        "Answer",
        "ConfigurationError",
        "Constraints",
        "EngineError",
        "GenerationError",
        "KnowledgeEngine",
        "KnowledgeError",
        "Provenance",
        "Query",
        "QueryError",
        "TimeoutError",
        "UnsupportedOperationError",
    ):
        from fitz_sage import core

        return getattr(core, name)

    # Runtime (heavy - discovers all engines)
    if name in (
        "create_engine",
        "get_engine_registry",
        "list_engines",
        "list_engines_with_info",
        "run",
    ):
        from fitz_sage import runtime

        return getattr(runtime, name)

    # SDK
    if name == "fitz":
        from fitz_sage import sdk

        return getattr(sdk, name)

    raise AttributeError(f"module 'fitz_sage' has no attribute {name!r}")


# =============================================================================
# MODULE-LEVEL SDK (matches CLI: fitz point, fitz query)
# =============================================================================

_default_fitz = None


def _get_default_fitz():
    """Get or create the default fitz instance."""
    global _default_fitz
    if _default_fitz is None:
        from fitz_sage.sdk import fitz

        _default_fitz = fitz()
    return _default_fitz


def query(question: str, source=None, collection: str = None, top_k: int = None):
    """
    Query the knowledge base.

    Module-level convenience function matching `fitz query` CLI.

    Args:
        question: The question to ask.
        source: Path to file or directory. If provided, registers documents
            before querying (equivalent to CLI --source flag).
        collection: Collection name (uses default if not specified).
        top_k: Number of chunks to retrieve (uses config default if not specified).

    Returns:
        Answer with text and provenance.

    Examples:
        >>> import fitz_sage
        >>> answer = fitz_sage.query("What is the refund policy?", source="./docs")
        >>> print(answer.text)
    """
    global _default_fitz
    if collection is not None:
        from fitz_sage.sdk import fitz

        _default_fitz = fitz(collection=collection)
    f = _get_default_fitz()
    return f.query(question, source=source, top_k=top_k)


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
    # SDK
    "fitz",
    "query",
]
