# fitz_sage/retrieval/detection/registry.py
"""
Detection summary — typed accessors over the per-module detection results.

The detection modules are classified in one batched LLM call by
``QueryBatcher``; ``DetectionSummary`` wraps the per-category results into
convenient accessors for retrieval routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from .modules import AggregationType, TemporalIntent
from .protocol import DetectionResult


@dataclass
class DetectionSummary:
    """
    Summary of all detections for retrieval routing.

    Provides convenient accessors for retrieval routing decisions.
    """

    temporal: DetectionResult[Any]
    aggregation: DetectionResult[Any]
    comparison: DetectionResult[Any]
    freshness: DetectionResult[Any]

    @property
    def has_temporal_intent(self) -> bool:
        """True if temporal intent detected."""
        return self.temporal.detected

    @property
    def has_aggregation_intent(self) -> bool:
        """True if aggregation intent detected."""
        return self.aggregation.detected

    @property
    def has_comparison_intent(self) -> bool:
        """True if comparison intent detected."""
        return self.comparison.detected

    @property
    def boost_recency(self) -> bool:
        """True if recency boosting should be applied."""
        return self.freshness.detected and self.freshness.metadata.get("boost_recency", False)

    @property
    def query_variations(self) -> list[str]:
        """Temporal sub-queries — one per detected time period."""
        if self.temporal.detected:
            return self.temporal.transformations
        return []

    @property
    def temporal_intent(self) -> TemporalIntent | None:
        """Get the temporal intent enum value."""
        if self.temporal.detected:
            return self.temporal.intent
        return None

    @property
    def aggregation_type(self) -> AggregationType | None:
        """Get the aggregation type enum value."""
        if self.aggregation.detected:
            return self.aggregation.intent
        return None

    @property
    def fetch_multiplier(self) -> int:
        """Get fetch multiplier from aggregation if detected."""
        if self.aggregation.detected:
            return int(self.aggregation.metadata.get("fetch_multiplier", 1))
        return 1

    @property
    def comparison_entities(self) -> list[str]:
        """Get entities being compared."""
        if self.comparison.detected:
            return cast(list[str], self.comparison.metadata.get("entities", []))
        return []

    @property
    def comparison_queries(self) -> list[str]:
        """Get expanded comparison queries."""
        if self.comparison.detected:
            return self.comparison.transformations
        return []
