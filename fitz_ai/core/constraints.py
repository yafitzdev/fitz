"""
Constraints - Query-time constraints for knowledge engines.

Constraints allow users to control query execution without changing engine configuration.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Constraints:
    """
    Query-time constraints (paradigm-agnostic).

    Constraints provide a way to customize query execution without modifying
    the engine's configuration. They are optional and engines should provide
    sensible defaults.

    Examples:
        Limit number of sources:
        >>> constraints = Constraints(max_sources=5)

        Filter by metadata:
        >>> constraints = Constraints(
        ...     filters={"topic": "quantum_physics", "year": 2023}
        ... )

        Combined constraints:
        >>> constraints = Constraints(
        ...     max_sources=10,
        ...     filters={"author": "Einstein"},
        ...     metadata={"timeout_seconds": 30}
        ... )
    """

    max_sources: Optional[int] = None
    """
    Maximum number of sources to use for answer generation.

    For Classic RAG: limits number of chunks retrieved
    For CLaRa: might limit number of documents consulted
    For other engines: interpret as makes sense

    If None, engine uses its default value.
    """

    filters: Dict[str, Any] = field(default_factory=dict)
    """
    Metadata filters to apply during knowledge selection.

    These are key-value pairs that sources must match to be considered.
    For example:
    - {"topic": "physics"} - only sources tagged with topic=physics
    - {"year": 2023} - only sources from 2023
    - {"author": "Smith", "reviewed": True} - AND conditions

    The exact semantics are engine-specific:
    - Classic RAG: applies as vector DB metadata filters
    - CLaRa: might filter document corpus
    - Custom engines: define their own filter logic

    Engines should ignore unknown filter keys gracefully.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Additional engine-specific constraint metadata.

    This allows passing constraints that don't fit the standard fields.
    Examples:
    - {"timeout_seconds": 30} - execution timeout
    - {"temperature": 0.3} - LLM sampling temperature
    - {"rerank": False} - disable reranking in Classic RAG

    Engines should ignore unknown metadata keys gracefully.
    """

    def __post_init__(self):
        """Validate constraints after initialization."""
        if self.max_sources is not None and self.max_sources < 1:
            raise ValueError("max_sources must be at least 1")
