# fitz_sage/retrieval/__init__.py
"""
Retrieval intelligence infrastructure.

This package contains query-time retrieval components:
- detection: LLM-based query classification and dict-based expansion
- entity_graph: Entity-based related chunk discovery
"""

from fitz_sage.retrieval.detection import (
    AggregationType,
    DetectionResult,
    DetectionSummary,
    TemporalIntent,
)
from fitz_sage.retrieval.entity_graph import EntityGraphStore

__all__ = [
    # Detection
    "AggregationType",
    "DetectionResult",
    "DetectionSummary",
    "TemporalIntent",
    # Entity graph
    "EntityGraphStore",
]
