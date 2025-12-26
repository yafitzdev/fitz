# fitz_ai/ingest/enrichment/summary/__init__.py
"""
Chunk-level summary generation.

This module handles generating LLM-based descriptions for individual chunks.
Summaries improve search relevance by providing natural language descriptions
that match how developers ask questions.

Components:
- Summarizer: Generates descriptions using LLM
- SummaryCache: Caches descriptions to avoid redundant API calls
"""

from fitz_ai.ingest.enrichment.summary.cache import (
    SummaryCache,
)
from fitz_ai.ingest.enrichment.summary.summarizer import (
    ChunkSummarizer,
)

__all__ = [
    "ChunkSummarizer",
    "SummaryCache",
]
