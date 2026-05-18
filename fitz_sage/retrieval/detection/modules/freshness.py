# fitz_sage/retrieval/detection/modules/freshness.py
"""Freshness detection module."""

from __future__ import annotations

from typing import Any

from fitz_sage.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DEFAULT_CONFIDENCE, DetectionModule


class FreshnessModule(DetectionModule):
    """Detects freshness signals — recency boosting."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.FRESHNESS

    @property
    def json_key(self) -> str:
        return "freshness"

    def prompt_fragment(self) -> str:
        return """"freshness": {
    "boost_recency": true/false
  }
  // boost_recency: "latest", "recent", "new", "current", "updated", "newest"
"""

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        boost_recency = data.get("boost_recency", False)

        if not boost_recency:
            return self.not_detected()

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=DEFAULT_CONFIDENCE,
            intent=None,
            matches=[],
            metadata={
                "boost_recency": boost_recency,
            },
            transformations=[],
        )
