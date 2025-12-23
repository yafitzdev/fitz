# fitz_ai/ingest/chunking/plugins/overlap.py
"""
Overlap chunker (DEPRECATED).

This chunker is deprecated - use SimpleChunker with chunk_overlap parameter instead.
Kept for backwards compatibility only.

Chunker ID format: "overlap:{chunk_size}:{chunk_overlap}"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List

from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass
class OverlapChunker:
    """
    Fixed-size chunker with overlap (DEPRECATED).

    DEPRECATED: Use SimpleChunker with chunk_overlap parameter instead.

    This class is kept for backwards compatibility. It delegates to the
    same logic as SimpleChunker.

    Example:
        >>> # Deprecated:
        >>> chunker = OverlapChunker(chunk_size=1000, chunk_overlap=100)
        >>> # Use instead:
        >>> chunker = SimpleChunker(chunk_size=1000, chunk_overlap=100)
    """

    plugin_name: str = field(default="overlap", repr=False)
    chunk_size: int = 1000
    chunk_overlap: int = 100

    def __post_init__(self) -> None:
        """Validate parameters and warn about deprecation."""
        warnings.warn(
            "OverlapChunker is deprecated. Use SimpleChunker with chunk_overlap parameter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

    @property
    def chunker_id(self) -> str:
        """Unique identifier for this chunker configuration."""
        return f"{self.plugin_name}:{self.chunk_size}:{self.chunk_overlap}"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: The text content to chunk.
            base_meta: Base metadata to include in each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        doc_id = str(
            base_meta.get("doc_id") or base_meta.get("source_file") or "unknown"
        )

        chunks: List[Chunk] = []
        chunk_index = 0

        step = self.chunk_size - self.chunk_overlap
        if step < 1:
            step = 1

        pos = 0
        while pos < len(text):
            end = pos + self.chunk_size
            piece = text[pos:end].strip()

            if piece:
                chunk_id = f"{doc_id}:{chunk_index}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        content=piece,
                        metadata=dict(base_meta),
                    )
                )
                chunk_index += 1

            pos += step

        return chunks


__all__ = ["OverlapChunker"]