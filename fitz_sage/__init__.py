# fitz_sage/__init__.py
"""
Fitz - Local-First Governed Retrieval & Engine Platform

Fitz is a paradigm-agnostic knowledge engine platform that supports multiple
approaches to knowledge retrieval, evidence packaging, and optional synthesis.

Quick Start:
    >>> from fitz_sage import evidence
    >>> pack = evidence("What is quantum computing?")
    >>> print(pack.mode)

Public API:
    Core Types:
        - Query: Input to engines
        - EvidencePack: Retrieval-first output from engines
        - Answer: Optional synthesized output from engines
        - Provenance: Source attribution

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
    Knowledge → Engine → EvidencePack → optional Answer

    Retrieval engines transform queries into governed evidence; the minimal
    KnowledgeEngine protocol returns an Answer. The runtime depends on those
    public contracts rather than engine internals.

Examples:
    Simple query:
    >>> from fitz_sage import evidence
    >>> pack = evidence("What is quantum computing?")

    Specific engine with answer synthesis configured:
    >>> answer = run("What is X?", engine="fitz_krag")  # requires synthesizer

    Reusable engine:
    >>> from fitz_sage import Query, create_engine
    >>> engine = create_engine("fitz_krag")
    >>> query = Query(text="What is Y?")
    >>> pack = engine.evidence(query)
"""

__version__ = "0.16.0"

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
        "EngineError",
        "GenerationError",
        "PyrrhoReplay",
        "EvidenceItem",
        "EvidencePack",
        "KnowledgeEngine",
        "KnowledgeError",
        "Provenance",
        "Query",
        "QueryError",
        "RetrievalRun",
        "RetrievalEngine",
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
        "load_retrieval_run",
        "replay_pyrrho",
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
# MODULE-LEVEL SDK (matches CLI: fitz point, fitz retrieve)
# =============================================================================

_default_fitz = None


def _get_default_fitz():
    """Get or create the default fitz instance."""
    global _default_fitz
    if _default_fitz is None:
        from fitz_sage.sdk import fitz

        _default_fitz = fitz()
    return _default_fitz


def answer(question: str, source=None, collection: str | None = None):
    """
    Synthesize an answer from retrieved evidence.

    Module-level convenience function matching `fitz answer` CLI.

    Args:
        question: The question to ask.
        source: Path to file or directory. If provided, registers documents
            before querying (equivalent to CLI --source flag).
        collection: Collection name (uses default if not specified).

    Returns:
        Answer with text and provenance.

    Examples:
        >>> import fitz_sage
        >>> answer = fitz_sage.answer("What is the refund policy?", source="./docs")
        >>> print(answer.text)
    """
    global _default_fitz
    if collection is not None:
        from fitz_sage.sdk import fitz

        _default_fitz = fitz(collection=collection)
    f = _get_default_fitz()
    return f.answer(question, source=source)


def evidence(question: str, source=None, collection: str | None = None):
    """
    Retrieve governed evidence without answer synthesis.

    Module-level convenience function matching `fitz retrieve` CLI.

    Args:
        question: The question to retrieve evidence for.
        source: Path to file or directory. If provided, registers documents
            before retrieving evidence (equivalent to CLI --source flag).
        collection: Collection name (uses default if not specified).

    Returns:
        EvidencePack with ranked source units, provenance, and governance mode.

    Examples:
        >>> import fitz_sage
        >>> pack = fitz_sage.evidence("What is the refund policy?", source="./docs")
        >>> print(pack.mode)
    """
    global _default_fitz
    if collection is not None:
        from fitz_sage.sdk import fitz

        _default_fitz = fitz(collection=collection)
    f = _get_default_fitz()
    return f.evidence(question, source=source)


def trace(question: str, source=None, collection: str | None = None):
    """
    Execute governed retrieval and return its versioned execution record.

    Serialization redacts source content by default. Call
    ``run.write(path, include_content=True)`` only when Pyrrho replay is
    required and the trace can be handled as source data.
    """
    global _default_fitz
    if collection is not None:
        from fitz_sage.sdk import fitz

        _default_fitz = fitz(collection=collection)
    f = _get_default_fitz()
    return f.trace(question, source=source)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core Protocol
    "KnowledgeEngine",
    "RetrievalEngine",
    # Core Types
    "Query",
    "Answer",
    "EvidenceItem",
    "EvidencePack",
    "PyrrhoReplay",
    "Provenance",
    "RetrievalRun",
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
    "load_retrieval_run",
    "replay_pyrrho",
    # SDK
    "answer",
    "fitz",
    "evidence",
    "trace",
]
