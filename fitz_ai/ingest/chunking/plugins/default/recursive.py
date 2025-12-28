# fitz_ai/ingest/chunking/plugins/default/recursive.py
"""
Recursive character text splitter.

The gold standard for "simple but good" chunking:
1. Try to split on paragraphs (\n\n)
2. If chunks still too big, split on lines (\n)
3. If still too big, split on sentences (. )
4. If still too big, split on words ( )
5. Last resort: split on characters

This preserves natural text boundaries while guaranteeing chunk size limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass
class RecursiveChunker:
    """
    Recursive character text splitter with overlap support.

    Splits text using a hierarchy of separators, trying to keep
    natural boundaries (paragraphs > lines > sentences > words).

    Args:
        chunk_size: Target chunk size in characters (default: 1000)
        chunk_overlap: Overlap between chunks in characters (default: 200)
        separators: List of separators to try, in order of preference
    """

    plugin_name: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", ", ", " ", ""])

    @property
    def chunker_id(self) -> str:
        """
        Unique identifier including parameters that affect chunk output.

        Format: "recursive:{chunk_size}:{chunk_overlap}"
        """
        return f"{self.plugin_name}:{self.chunk_size}:{self.chunk_overlap}"

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using the separator hierarchy.
        """
        if not text:
            return []

        # Base case: text fits in chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Try each separator in order
        for i, sep in enumerate(separators):
            if sep == "":
                # Last resort: hard character split
                return self._hard_split(text)

            if sep not in text:
                continue

            # Split on this separator
            parts = text.split(sep)

            # Merge small parts back together
            chunks = []
            current = ""

            for part in parts:
                # Add separator back (except for last part)
                part_with_sep = part + sep if part != parts[-1] else part

                if not current:
                    current = part_with_sep
                elif len(current) + len(part_with_sep) <= self.chunk_size:
                    current += part_with_sep
                else:
                    # Current chunk is full
                    if current.strip():
                        chunks.append(current.strip())

                    # Check if this part alone is too big
                    if len(part_with_sep) > self.chunk_size:
                        # Recursively split with remaining separators
                        sub_chunks = self._split_text(part_with_sep, separators[i + 1 :])
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part_with_sep

            # Don't forget the last chunk
            if current.strip():
                # Check if it's too big
                if len(current) > self.chunk_size:
                    sub_chunks = self._split_text(current, separators[i + 1 :])
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(current.strip())

            return chunks

        # No separator worked, hard split
        return self._hard_split(text)

    def _hard_split(self, text: str) -> List[str]:
        """
        Last resort: split by character count.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """
        Add overlap between chunks.

        Takes the last N characters from the previous chunk
        and prepends to the current chunk.
        """
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap :]

            # Find a clean break point (prefer word boundary)
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1 :]

            # Prepend overlap to current chunk
            combined = overlap_text + " " + curr_chunk
            result.append(combined.strip())

        return result

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk text using recursive splitting with overlap.

        Args:
            text: Raw text to chunk
            base_meta: Base metadata (doc_id, source_file, etc.)

        Returns:
            List of Chunk objects
        """
        doc_id = str(base_meta.get("doc_id") or base_meta.get("source_file") or "unknown")

        # Split text
        raw_chunks = self._split_text(text, self.separators)

        # Add overlap
        chunks_with_overlap = self._add_overlap(raw_chunks)

        # Build Chunk objects
        chunks: List[Chunk] = []
        for i, content in enumerate(chunks_with_overlap):
            if not content.strip():
                continue

            chunk_id = f"{doc_id}:{i}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=i,
                    content=content,
                    metadata=dict(base_meta),
                )
            )

        return chunks
