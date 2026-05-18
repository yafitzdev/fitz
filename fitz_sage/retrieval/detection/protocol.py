# fitz_sage/retrieval/detection/protocol.py
"""
Base protocols and result types for unified detection system.

Detection is now LLM-based, so this module only contains the data types
used for results and matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class DetectionCategory(Enum):
    """Categories of query detection."""

    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    FRESHNESS = "freshness"


@dataclass
class Match:
    """
    A single pattern match result.

    Attributes:
        text: The matched text from the query
        pattern_name: Name of the pattern that matched
        start: Start position in query
        end: End position in query
        groups: Named groups from regex (if applicable)
        confidence: Match confidence (0.0-1.0)
    """

    text: str
    pattern_name: str
    start: int = 0
    end: int = 0
    groups: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class DetectionResult(Generic[T]):
    """
    Unified result from detection.

    Attributes:
        detected: Whether a detection was made
        category: The detection category
        confidence: Detection confidence (0.0-1.0)
        intent: Optional intent enum value (e.g., TemporalIntent.COMPARISON)
        matches: List of individual matches
        metadata: Category-specific metadata
        transformations: Suggested query transformations
    """

    detected: bool
    category: DetectionCategory
    confidence: float = 0.0
    intent: T | None = None
    matches: list[Match] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    transformations: list[str] = field(default_factory=list)

    @classmethod
    def not_detected(cls, category: DetectionCategory) -> DetectionResult[Any]:
        """Create a not-detected result."""
        return cls(detected=False, category=category)
