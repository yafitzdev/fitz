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
    fitz_ai/
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
        from fitz_ai import core

        return getattr(core, name)

    # Classic RAG engine (heavy - loads config, plugins)
    if name in ("ClassicRagEngine", "create_classic_rag_engine", "run_classic_rag"):
        from fitz_ai.engines import classic_rag

        return getattr(classic_rag, name)

    # Runtime (heavy - discovers all engines)
    if name in ("create_engine", "get_engine_registry", "list_engines", "list_engines_with_info", "run"):
        from fitz_ai import runtime

        return getattr(runtime, name)

    # SDK
    if name in ("IngestStats", "fitz"):
        from fitz_ai import sdk

        return getattr(sdk, name)

    raise AttributeError(f"module 'fitz_ai' has no attribute {name!r}")

# =============================================================================
# MODULE-LEVEL SDK (matches CLI: fitz ingest, fitz query)
# =============================================================================

_default_fitz = None


def _get_default_fitz():
    """Get or create the default fitz instance."""
    global _default_fitz
    if _default_fitz is None:
        from fitz_ai.sdk import fitz

        _default_fitz = fitz()
    return _default_fitz


def ingest(source, collection: str = None, clear_existing: bool = False):
    """
    Ingest documents into the knowledge base.

    Module-level convenience function matching `fitz ingest` CLI.

    Args:
        source: Path to file or directory to ingest.
        collection: Collection name (uses default if not specified).
        clear_existing: If True, clear collection before ingesting.

    Returns:
        IngestStats with document and chunk counts.

    Examples:
        >>> import fitz_ai
        >>> fitz_ai.ingest("./docs")
        >>> answer = fitz_ai.query("What is X?")
    """
    global _default_fitz
    if collection is not None:
        from fitz_ai.sdk import fitz

        _default_fitz = fitz(collection=collection)
    f = _get_default_fitz()
    return f.ingest(source, clear_existing=clear_existing)


def query(question: str, top_k: int = None):
    """
    Query the knowledge base.

    Module-level convenience function matching `fitz query` CLI.

    Args:
        question: The question to ask.
        top_k: Number of chunks to retrieve (uses config default if not specified).

    Returns:
        Answer with text and provenance.

    Examples:
        >>> import fitz_ai
        >>> fitz_ai.ingest("./docs")
        >>> answer = fitz_ai.query("What is the refund policy?")
        >>> print(answer.text)
    """
    f = _get_default_fitz()
    return f.ask(question, top_k=top_k)


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
    # SDK
    "fitz",
    "IngestStats",
    "ingest",
    "query",
]
