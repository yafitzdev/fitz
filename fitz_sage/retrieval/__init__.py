# fitz_sage/retrieval/__init__.py
"""
Retrieval intelligence infrastructure.

This package contains query-time retrieval components:
- detection: LLM-based query classification and dict-based expansion
- sparse: BM25/TF-IDF sparse index for hybrid search
- entity_graph: Entity-based related chunk discovery
- vocabulary: Keyword vocabulary for exact matching
"""

from fitz_sage.retrieval.detection import (
    AggregationType,
    DetectionOrchestrator,
    DetectionResult,
    DetectionSummary,
    TemporalIntent,
)
from fitz_sage.retrieval.entity_graph import EntityGraphStore
from fitz_sage.retrieval.vocabulary import (
    Keyword,
    KeywordDetector,
    KeywordMatcher,
    VocabularyStore,
    create_matcher_from_store,
    generate_variations,
)

__all__ = [
    # Detection
    "AggregationType",
    "DetectionOrchestrator",
    "DetectionResult",
    "DetectionSummary",
    "TemporalIntent",
    # Entity graph
    "EntityGraphStore",
    # Vocabulary
    "Keyword",
    "KeywordDetector",
    "KeywordMatcher",
    "VocabularyStore",
    "create_matcher_from_store",
    "generate_variations",
]
