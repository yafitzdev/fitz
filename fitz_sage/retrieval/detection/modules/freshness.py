# fitz_sage/retrieval/detection/modules/freshness.py
"""Freshness detection module."""

from __future__ import annotations

from typing import Any

from fitz_sage.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .base import DEFAULT_CONFIDENCE, DetectionModule


class FreshnessModule(DetectionModule):
    """Detect queries asking for current or recent evidence."""

    @property
    def category(self) -> DetectionCategory:
        return DetectionCategory.FRESHNESS

    @property
    def json_key(self) -> str:
        return "freshness"

    def prompt_fragment(self) -> str:
        return """"freshness": {
    "detected": true/false
  }
  // freshness: "latest", "recent", "new", "current", "updated", "newest"
"""

    def parse_result(self, data: dict[str, Any]) -> DetectionResult[None]:
        if not data.get("detected", False):
            return self.not_detected()

        return DetectionResult(
            detected=True,
            category=self.category,
            confidence=DEFAULT_CONFIDENCE,
            intent=None,
            matches=[],
            metadata={},
            transformations=[],
        )
