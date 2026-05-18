# fitz_sage/retrieval/detection/__init__.py
"""
Detection system — module-based LLM query classification.

Each module defines its prompt fragment and parsing logic; all modules are
combined into the single batched query-prep LLM call (``QueryBatcher``), and
``DetectionSummary`` wraps the per-category results for retrieval routing.

To add a new detection category:
1. Create a module in modules/ (inherit from DetectionModule)
2. Implement category, json_key, prompt_fragment(), parse_result()
3. Add to DEFAULT_MODULES in modules/__init__.py
"""

from .modules import (
    DEFAULT_MODULES,
    AggregationModule,
    AggregationType,
    ComparisonModule,
    DetectionModule,
    FreshnessModule,
    TemporalIntent,
    TemporalModule,
    distribute_to_modules,
)
from .protocol import DetectionCategory, DetectionResult, Match
from .registry import DetectionSummary

__all__ = [
    "DetectionCategory",
    "DetectionResult",
    "DetectionSummary",
    "Match",
    # Enums
    "AggregationType",
    "TemporalIntent",
    # Module system
    "DEFAULT_MODULES",
    "DetectionModule",
    "distribute_to_modules",
    # Modules
    "AggregationModule",
    "ComparisonModule",
    "FreshnessModule",
    "TemporalModule",
]
