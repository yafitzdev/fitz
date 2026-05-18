# fitz_sage/engines/fitz_krag/query_analyzer.py
"""
Query analysis types for fitz_krag — knowledge-type classification.

Query classification runs as a section of the batched query-prep call
(``QueryBatcher``). This module holds the shared result types, the per-type
strategy weights, and the dict parser the batcher uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    """Knowledge type a query targets."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    GENERAL = "general"
    CROSS = "cross"
    DATA = "data"


@dataclass(frozen=True)
class QueryAnalysis:
    """Result of analyzing a query's knowledge type intent."""

    primary_type: QueryType
    secondary_type: QueryType | None = None
    confidence: float = 0.5
    entities: tuple[str, ...] = ()
    refined_query: str = ""

    @property
    def strategy_weights(self) -> dict[str, float]:
        """Compute per-strategy weights from query analysis."""
        weights = _TYPE_WEIGHTS.get(self.primary_type, _TYPE_WEIGHTS[QueryType.GENERAL])
        return dict(weights)


# Default strategy weights per query type
_TYPE_WEIGHTS: dict[QueryType, dict[str, float]] = {
    QueryType.CODE: {"code": 0.75, "section": 0.1, "table": 0.05, "chunk": 0.1},
    QueryType.DOCUMENTATION: {"code": 0.1, "section": 0.75, "table": 0.05, "chunk": 0.1},
    QueryType.GENERAL: {"code": 0.25, "section": 0.25, "table": 0.15, "chunk": 0.35},
    QueryType.CROSS: {"code": 0.35, "section": 0.35, "table": 0.1, "chunk": 0.2},
    QueryType.DATA: {"code": 0.05, "section": 0.15, "table": 0.70, "chunk": 0.10},
}


def parse_analysis_dict(data: dict, original_query: str) -> QueryAnalysis:
    """Parse a JSON-decoded analysis dict into QueryAnalysis.

    Used by the ``QueryBatcher`` analysis section.
    """
    primary = _parse_query_type(data.get("primary_type") or "general")
    secondary = None
    if data.get("secondary_type"):
        secondary = _parse_query_type(data["secondary_type"])

    entities = data.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    return QueryAnalysis(
        primary_type=primary,
        secondary_type=secondary,
        confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
        entities=tuple(str(e) for e in entities),
        refined_query=str(data.get("refined_query", original_query)),
    )


def _parse_query_type(value: str) -> QueryType:
    """Parse a string into QueryType, defaulting to GENERAL."""
    try:
        return QueryType(value.lower())
    except ValueError:
        return QueryType.GENERAL
