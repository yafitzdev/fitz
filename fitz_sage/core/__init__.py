# fitz_sage/core/__init__.py
"""
Fitz Core - Paradigm-agnostic contracts for knowledge engines.

This module defines the stable abstractions that all engines must implement.
The core philosophy is: Knowledge → Engine → Answer.

Public API:
    - KnowledgeEngine: Protocol that all engines implement
    - Query: Input to engines
    - Answer: Output from engines
    - Provenance: Source attribution
    - FitzPaths: Central path management
    - Exceptions: Standard error hierarchy

Examples:
    Using the core abstractions:
    >>> from fitz_sage.core import Query
    >>> from fitz_sage.engines.fitz_krag import FitzKragEngine
    >>>
    >>> # Create an engine (engine-specific)
    >>> engine = FitzKragEngine(config)
    >>>
    >>> # Use core abstractions (paradigm-agnostic)
    >>> query = Query(text="What is quantum computing?")
    >>> answer = engine.answer(query)
    >>>
    >>> # Access results (paradigm-agnostic)
    >>> print(answer.text)
    >>> for source in answer.provenance:
    ...     print(f"Source: {source.source_id}")

    Using FitzPaths:
    >>> from fitz_sage.core import FitzPaths
    >>> config_path = FitzPaths.config()

"""

from .answer import Answer
from .collections import collection_name_from_path, validate_collection_name

# Core protocol
from .engine import KnowledgeEngine, RetrievalEngine
from .evidence import EvidenceItem, EvidencePack

# Core exceptions
from .exceptions import (
    ConfigurationError,
    EngineError,
    EnrichmentError,
    GenerationError,
    KnowledgeError,
    ManagedModelError,
    QueryError,
    QueryIntelligenceError,
    TimeoutError,
    UnsupportedOperationError,
)

# Path management
from .paths import FitzPaths, get_config_path, get_workspace
from .provenance import Provenance

# Core types
from .query import Query
from .retrieval_run import (
    RETRIEVAL_RUN_SCHEMA_VERSION,
    CandidateReference,
    CandidateStage,
    FrozenEvidence,
    GovernanceExecution,
    GovernanceReplay,
    GovernanceStep,
    QueryExecution,
    QueryTerm,
    RetrievalRun,
    RunEnvironment,
    StrategyExecution,
)

__all__ = [
    # Protocol
    "KnowledgeEngine",
    "RetrievalEngine",
    # Types
    "Query",
    "Answer",
    "EvidenceItem",
    "EvidencePack",
    "RETRIEVAL_RUN_SCHEMA_VERSION",
    "CandidateReference",
    "CandidateStage",
    "FrozenEvidence",
    "GovernanceExecution",
    "GovernanceReplay",
    "GovernanceStep",
    "QueryExecution",
    "QueryTerm",
    "RetrievalRun",
    "RunEnvironment",
    "StrategyExecution",
    "Provenance",
    "validate_collection_name",
    "collection_name_from_path",
    # Path Management
    "FitzPaths",
    "get_workspace",
    "get_config_path",
    # Exceptions
    "EngineError",
    "QueryError",
    "KnowledgeError",
    "GenerationError",
    "QueryIntelligenceError",
    "EnrichmentError",
    "ManagedModelError",
    "ConfigurationError",
    "TimeoutError",
    "UnsupportedOperationError",
]
