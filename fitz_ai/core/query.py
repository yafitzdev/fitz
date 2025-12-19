"""
Query - Paradigm-agnostic query representation.

A Query encapsulates everything needed to ask a question to a knowledge engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .constraints import Constraints


@dataclass
class Query:
    """
    Paradigm-agnostic query representation.

    A query contains:
    - text: The question being asked
    - constraints: Optional query-time constraints (filters, limits, etc.)
    - metadata: Engine-specific hints and configuration

    Examples:
        Simple query:
        >>> query = Query(text="What is quantum computing?")

        Query with constraints:
        >>> constraints = Constraints(max_sources=5, filters={"topic": "physics"})
        >>> query = Query(text="Explain entanglement", constraints=constraints)

        Query with engine hints:
        >>> query = Query(
        ...     text="Summarize the paper",
        ...     metadata={"temperature": 0.3, "model": "claude-3-opus"}
        ... )
    """

    text: str
    """The question or instruction text."""

    constraints: Optional[Constraints] = None
    """Optional query-time constraints (filters, source limits, etc.)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Engine-specific hints and configuration.

    This allows passing engine-specific parameters without breaking the
    paradigm-agnostic interface. For example:
    - Classic RAG might use: {"rerank": True, "top_k": 10}
    - CLaRa might use: {"uncertainty_threshold": 0.3}
    - Future engines can define their own metadata keys

    Engines should ignore unknown metadata keys gracefully.
    """

    def __post_init__(self):
        """Validate query after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")
