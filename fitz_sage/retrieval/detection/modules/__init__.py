# fitz_sage/retrieval/detection/modules/__init__.py
"""
Detection modules for LLM-based query classification.

Each module contributes a prompt fragment and parsing logic; all modules are
combined into the single batched query-prep LLM call.

To add a new detection category:
1. Create a new module file (e.g., my_category.py)
2. Inherit from DetectionModule
3. Implement category, json_key, prompt_fragment(), parse_result()
4. Add to DEFAULT_MODULES below
"""

from __future__ import annotations

from typing import Any

from fitz_sage.retrieval.detection.protocol import DetectionCategory, DetectionResult

from .aggregation import AggregationModule, AggregationType
from .base import DetectionModule
from .comparison import ComparisonModule
from .freshness import FreshnessModule
from .temporal import TemporalIntent, TemporalModule

# Default modules — combined into the single batched query-prep LLM call.
DEFAULT_MODULES: list[DetectionModule] = [
    TemporalModule(),
    AggregationModule(),
    ComparisonModule(),
    FreshnessModule(),
]


def distribute_to_modules(
    raw_results: dict[str, Any],
    modules: list[DetectionModule],
) -> dict[DetectionCategory, DetectionResult[Any]]:
    """Distribute parsed JSON results to detection modules for parsing."""
    results = {}
    for module in modules:
        module_data = raw_results.get(module.json_key, {})
        if not isinstance(module_data, dict):
            module_data = {}
        results[module.category] = module.parse_result(module_data)
    return results


__all__ = [
    "DetectionModule",
    "DEFAULT_MODULES",
    "distribute_to_modules",
    "AggregationModule",
    "ComparisonModule",
    "FreshnessModule",
    "TemporalModule",
    "AggregationType",
    "TemporalIntent",
]
