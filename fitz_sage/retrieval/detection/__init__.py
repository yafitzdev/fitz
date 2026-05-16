# fitz_sage/retrieval/detection/__init__.py
"""
Unified detection system using module-based LLM classification.

Similar to the enrichment bus pattern:
- Each module defines its prompt fragment and parsing logic
- All modules are combined into a single LLM call
- Results are distributed to modules for parsing

To add a new detection category:
1. Create a module in modules/ (inherit from DetectionModule)
2. Implement category, json_key, prompt_fragment(), parse_result()
3. Add to DEFAULT_MODULES in modules/__init__.py

Usage:
    from fitz_sage.retrieval.detection import DetectionOrchestrator
    from fitz_sage.llm import get_chat_factory

    factory = get_chat_factory("cohere")
    orchestrator = DetectionOrchestrator(chat_factory=factory)
    summary = orchestrator.detect_for_retrieval(query)

    if summary.has_temporal_intent:
        # Route to temporal strategy
        ...
"""

from .llm_classifier import LLMClassifier
from .modules import (
    DEFAULT_MODULES,
    AggregationModule,
    AggregationType,
    ComparisonModule,
    DetectionModule,
    FreshnessModule,
    RewriterModule,
    TemporalIntent,
    TemporalModule,
)
from .protocol import DetectionCategory, DetectionResult, Match
from .registry import DetectionOrchestrator, DetectionSummary

__all__ = [
    # Orchestration
    "DetectionCategory",
    "DetectionOrchestrator",
    "DetectionResult",
    "DetectionSummary",
    # Enums
    "AggregationType",
    "TemporalIntent",
    # Protocol
    "Match",
    # LLM Classifier
    "LLMClassifier",
    # Module system
    "DEFAULT_MODULES",
    "DetectionModule",
    # Modules
    "AggregationModule",
    "ComparisonModule",
    "FreshnessModule",
    "RewriterModule",
    "TemporalModule",
]
