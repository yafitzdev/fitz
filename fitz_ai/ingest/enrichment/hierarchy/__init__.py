# fitz_ai/ingest/enrichment/hierarchy/__init__.py
"""
Hierarchical enrichment module.

Provides multi-level summarization based on configurable grouping rules.
Users configure rules in YAML to group chunks by metadata keys and
generate summaries at group and corpus levels.

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

from fitz_ai.ingest.enrichment.hierarchy.enricher import HierarchyEnricher
from fitz_ai.ingest.enrichment.hierarchy.grouper import ChunkGrouper
from fitz_ai.ingest.enrichment.hierarchy.matcher import ChunkMatcher

__all__ = ["HierarchyEnricher", "ChunkMatcher", "ChunkGrouper"]
