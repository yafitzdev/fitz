# fitz_sage/retrieval/detection/registry.py
"""
Detection orchestrator using module-based LLM classification.

Similar to the enrichment bus pattern - each module contributes its
prompt fragment and parsing logic, but all are combined into one LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fitz_sage.llm.factory import ChatFactory
from fitz_sage.logging.logger import get_logger

from .llm_classifier import LLMClassifier
from .modules import AggregationType, TemporalIntent
from .protocol import DetectionCategory, DetectionResult

logger = get_logger(__name__)


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
    rewriter: DetectionResult[Any]
    vocabulary: DetectionResult[Any]

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
    def boost_authority(self) -> bool:
        """True if authority boosting should be applied."""
        return self.freshness.detected and self.freshness.metadata.get("boost_authority", False)

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
            return self.aggregation.metadata.get("fetch_multiplier", 1)
        return 1

    @property
    def needs_rewriting(self) -> bool:
        """True if query might benefit from rewriting."""
        if not self.rewriter.detected:
            return False
        return self.rewriter.metadata.get("needs_context", False) or self.rewriter.metadata.get(
            "is_compound", False
        )

    @property
    def comparison_entities(self) -> list[str]:
        """Get entities being compared."""
        if self.comparison.detected:
            return self.comparison.metadata.get("entities", [])
        return []

    @property
    def comparison_queries(self) -> list[str]:
        """Get expanded comparison queries."""
        if self.comparison.detected:
            return self.comparison.transformations
        return []


@dataclass
class DetectionOrchestrator:
    """
    Orchestrates detection using module-based LLM classification.

    Similar to the enrichment bus - modules define prompt fragments and
    parsing, but all are combined into a single LLM call.

    Usage:
        from fitz_sage.llm import get_chat_factory

        factory = get_chat_factory("cohere")
        orchestrator = DetectionOrchestrator(chat_factory=factory)
        summary = orchestrator.detect_for_retrieval(query)

        if summary.has_temporal_intent:
            # Route to temporal strategy
            ...
    """

    chat_factory: ChatFactory | None = None

    # Lazy-loaded LLM classifier
    _classifier: LLMClassifier | None = field(default=None, init=False, repr=False)

    def _ensure_classifier(self) -> LLMClassifier | None:
        """Lazy-load LLM classifier with all modules."""
        if self._classifier is None and self.chat_factory is not None:
            self._classifier = LLMClassifier(chat_factory=self.chat_factory)
        return self._classifier

    def detect_for_retrieval(self, query: str) -> DetectionSummary:
        """
        Run detection optimized for retrieval routing.

        Uses module-based LLM classification for robust detection,
        with dict-based expansion for query variations.

        Args:
            query: User's query string

        Returns:
            DetectionSummary with routing information
        """
        classifier = self._ensure_classifier()

        if classifier:
            results = classifier.classify(query)
        else:
            results = {}
            logger.debug("No chat factory available, skipping LLM classification")

        return DetectionSummary(
            temporal=results.get(
                DetectionCategory.TEMPORAL,
                DetectionResult.not_detected(DetectionCategory.TEMPORAL),
            ),
            aggregation=results.get(
                DetectionCategory.AGGREGATION,
                DetectionResult.not_detected(DetectionCategory.AGGREGATION),
            ),
            comparison=results.get(
                DetectionCategory.COMPARISON,
                DetectionResult.not_detected(DetectionCategory.COMPARISON),
            ),
            freshness=results.get(
                DetectionCategory.FRESHNESS,
                DetectionResult.not_detected(DetectionCategory.FRESHNESS),
            ),
            rewriter=results.get(
                DetectionCategory.REWRITER,
                DetectionResult.not_detected(DetectionCategory.REWRITER),
            ),
            vocabulary=DetectionResult.not_detected(DetectionCategory.VOCABULARY),
        )
