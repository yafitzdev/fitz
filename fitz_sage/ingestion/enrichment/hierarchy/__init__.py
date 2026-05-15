# fitz_sage/ingestion/enrichment/hierarchy/__init__.py
"""
Hierarchical enrichment module.

Provides multi-level summarization based on configurable grouping rules.
Users configure rules in YAML to group chunks by metadata keys and
generate summaries at group and corpus levels.

Features epistemic-aware summarization that:
- Detects conflicts within chunk groups
- Adapts prompts to acknowledge uncertainty
- Propagates epistemic metadata up the hierarchy

Usage:
    enrichment:
      hierarchy:
        enabled: true
        rules:
          - name: video_comments
            paths: ["comments/**"]
            group_by: video_id
            prompt: "Summarize sentiment and themes"
"""

from fitz_sage.ingestion.enrichment.hierarchy.enricher import (
    EpistemicAssessment,
    HierarchyEnricher,
    assess_chunk_group,
)
from fitz_sage.ingestion.enrichment.hierarchy.grouper import ChunkGrouper
from fitz_sage.ingestion.enrichment.hierarchy.matcher import ChunkMatcher

__all__ = [
    "HierarchyEnricher",
    "ChunkMatcher",
    "ChunkGrouper",
    "EpistemicAssessment",
    "assess_chunk_group",
]
