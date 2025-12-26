# fitz_ai/ingest/enrichment/models.py
"""
Data models for the enrichment pipeline.

These models define the input/output contract for the enrichment "box":
- Input: List[Chunk]
- Output: EnrichmentResult (enriched chunks + artifacts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk
    from fitz_ai.ingest.enrichment.artifacts.base import Artifact


@dataclass
class EnrichmentResult:
    """
    Output of the enrichment pipeline.

    Contains both the enriched chunks (with summaries attached to metadata)
    and any corpus-level artifacts generated.

    Attributes:
        chunks: Chunks with enrichment data in metadata (e.g., metadata["summary"])
        artifacts: Corpus-level artifacts (navigation index, interface catalog, etc.)
    """

    chunks: List["Chunk"]
    artifacts: List["Artifact"]


__all__ = ["EnrichmentResult"]
