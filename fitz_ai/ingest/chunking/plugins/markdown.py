# fitz_ai/ingest/chunking/plugins/markdown.py
"""
Markdown-aware chunker.

Splits markdown documents on headers while preserving:
- Code blocks (fenced and indented)
- List structure
- Blockquotes

Also extracts YAML frontmatter and merges it into chunk metadata.

Chunker ID format: "markdown:{max_chunk_size}:{min_chunk_size}"
Example: "markdown:1500:100"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from fitz_ai.engines.classic_rag.models.chunk import Chunk

logger = logging.getLogger(__name__)


@dataclass
class MarkdownChunker:
    """
    Markdown-aware chunker that splits on headers.

    Strategy:
    1. Split on headers (# ## ### etc.)
    2. Each section becomes a chunk with header as context
    3. Large sections are split at paragraph boundaries
    4. Code blocks are never split mid-block
    5. Small adjacent sections may be merged

    Example:
        >>> chunker = MarkdownChunker(max_chunk_size=1500)
        >>> chunker.chunker_id
        'markdown:1500:100'
    """

    plugin_name: str = field(default="markdown", repr=False)
    supported_extensions: list[str] = field(
        default_factory=lambda: [".md", ".markdown", ".mdx"], repr=False
    )
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    include_header_in_chunk: bool = True

    # Regex patterns
    _HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    _FENCED_CODE_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _INDENTED_CODE_PATTERN = re.compile(r"(?:^(?:    |\t).+\n?)+", re.MULTILINE)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.max_chunk_size < 100:
            raise ValueError(
                f"max_chunk_size must be >= 100, got {self.max_chunk_size}"
            )
        if self.min_chunk_size < 1:
            raise ValueError(f"min_chunk_size must be >= 1, got {self.min_chunk_size}")
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) must be < max_chunk_size ({self.max_chunk_size})"
            )

    @property
    def chunker_id(self) -> str:
        """Unique identifier for this chunker configuration."""
        return f"{self.plugin_name}:{self.max_chunk_size}:{self.min_chunk_size}"

    def _find_code_blocks(self, text: str) -> List[Tuple[int, int]]:
        """Find all code block ranges (start, end) to avoid splitting them."""
        ranges = []

        # Fenced code blocks (```...```)
        for match in self._FENCED_CODE_PATTERN.finditer(text):
            ranges.append((match.start(), match.end()))

        # Indented code blocks (4 spaces or tab)
        for match in self._INDENTED_CODE_PATTERN.finditer(text):
            ranges.append((match.start(), match.end()))

        return sorted(ranges)

    def _is_in_code_block(self, pos: int, code_ranges: List[Tuple[int, int]]) -> bool:
        """Check if position is inside a code block."""
        for start, end in code_ranges:
            if start <= pos < end:
                return True
        return False

    def _split_into_sections(self, text: str) -> List[Tuple[Optional[str], str, int]]:
        """
        Split markdown into sections by headers.

        Returns list of (header_text, section_content, header_level).
        """
        code_ranges = self._find_code_blocks(text)
        sections: List[Tuple[Optional[str], str, int]] = []

        # Find all headers that are NOT inside code blocks
        header_positions = []
        for match in self._HEADER_PATTERN.finditer(text):
            if not self._is_in_code_block(match.start(), code_ranges):
                level = len(match.group(1))
                header_text = match.group(2).strip()
                header_positions.append(
                    (match.start(), match.end(), header_text, level)
                )

        if not header_positions:
            # No headers found, return entire text as one section
            return [(None, text.strip(), 0)]

        # Extract content before first header
        first_header_start = header_positions[0][0]
        if first_header_start > 0:
            preamble = text[:first_header_start].strip()
            if preamble:
                sections.append((None, preamble, 0))

        # Extract each section
        for i, (start, end, header_text, level) in enumerate(header_positions):
            # Section content is from after this header to next header (or end)
            if i + 1 < len(header_positions):
                next_start = header_positions[i + 1][0]
                content = text[end:next_start].strip()
            else:
                content = text[end:].strip()

            sections.append((header_text, content, level))

        return sections

    def _split_large_section(
        self,
        header: Optional[str],
        content: str,
        level: int,
    ) -> List[Tuple[Optional[str], str]]:
        """
        Split a large section into smaller chunks.

        Splits at paragraph boundaries, preserving code blocks.
        """
        # Build the full section text
        if header and self.include_header_in_chunk:
            prefix = "#" * max(level, 1) + " " + header + "\n\n"
        else:
            prefix = ""

        full_text = prefix + content

        if len(full_text) <= self.max_chunk_size:
            return [(header, full_text)]

        # Need to split - find paragraph boundaries
        chunks: List[Tuple[Optional[str], str]] = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\n+", content)

        current_chunks: List[str] = []
        current_size = len(prefix)
        part_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If single paragraph is too large, we have to include it anyway
            # (could be a code block)
            if para_size > self.max_chunk_size:
                # Flush current
                if current_chunks:
                    chunk_text = prefix + "\n\n".join(current_chunks)
                    chunk_header = (
                        f"{header} (Part {part_num})"
                        if header and part_num > 1
                        else header
                    )
                    chunks.append((chunk_header, chunk_text.strip()))
                    part_num += 1
                    current_chunks = []
                    current_size = len(prefix)

                # Add large paragraph as its own chunk
                chunk_header = f"{header} (Part {part_num})" if header else None
                chunk_text = prefix + para if part_num == 1 else para
                chunks.append((chunk_header, chunk_text.strip()))
                part_num += 1

            elif current_size + para_size + 2 > self.max_chunk_size and current_chunks:
                # Flush current chunk
                chunk_text = prefix + "\n\n".join(current_chunks)
                chunk_header = (
                    f"{header} (Part {part_num})" if header and part_num > 1 else header
                )
                chunks.append((chunk_header, chunk_text.strip()))
                part_num += 1
                current_chunks = [para]
                current_size = len(prefix) + para_size

            else:
                current_chunks.append(para)
                current_size += para_size + 2

        # Flush remaining
        if current_chunks:
            chunk_text = (prefix if part_num == 1 else "") + "\n\n".join(current_chunks)
            chunk_header = (
                f"{header} (Part {part_num})" if header and part_num > 1 else header
            )
            chunks.append((chunk_header, chunk_text.strip()))

        return chunks

    def _merge_small_sections(
        self,
        sections: List[Tuple[Optional[str], str, int]],
    ) -> List[Tuple[Optional[str], str, int]]:
        """Merge very small adjacent sections only when both are tiny."""
        if not sections or len(sections) <= 1:
            return sections

        merged: List[Tuple[Optional[str], str, int]] = []
        current_header, current_content, current_level = sections[0]

        for header, content, level in sections[1:]:
            # Only merge if BOTH are very small (less than min_chunk_size)
            # and the current one has no header (preamble)
            if (
                current_header is None
                and len(current_content) < self.min_chunk_size
                and len(content) < self.min_chunk_size
            ):
                # Merge preamble into next section
                if header and self.include_header_in_chunk:
                    header_prefix = "#" * max(level, 1) + " " + header + "\n\n"
                    current_content = current_content + "\n\n" + header_prefix + content
                else:
                    current_content = current_content + "\n\n" + content
                current_header = header
                current_level = level
            else:
                merged.append((current_header, current_content, current_level))
                current_header, current_content, current_level = header, content, level

        merged.append((current_header, current_content, current_level))
        return merged

    def _extract_frontmatter(self, text: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract YAML frontmatter from markdown text if present.

        Frontmatter is a YAML block at the start of the file delimited by ---:
            ---
            video_id: abc123
            tags: [tutorial, python]
            ---
            # Actual content starts here

        Args:
            text: Markdown text that may contain frontmatter.

        Returns:
            Tuple of (frontmatter_dict, remaining_content).
            If no frontmatter, returns ({}, original_text).
        """
        if not text.startswith("---"):
            return {}, text

        # Split on --- delimiter
        parts = text.split("---", 2)
        if len(parts) < 3:
            # Malformed frontmatter, return original
            return {}, text

        frontmatter_str = parts[1].strip()
        content = parts[2].lstrip("\n")

        if not frontmatter_str:
            return {}, content

        try:
            frontmatter = yaml.safe_load(frontmatter_str)
            if not isinstance(frontmatter, dict):
                # Frontmatter should be a dict, not a scalar or list
                logger.debug(
                    f"Frontmatter is not a dict, ignoring: {type(frontmatter)}"
                )
                return {}, text
            return frontmatter, content
        except yaml.YAMLError as e:
            logger.debug(f"Failed to parse frontmatter: {e}")
            return {}, text

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk markdown text by headers.

        Extracts YAML frontmatter and merges it into chunk metadata.

        Args:
            text: Markdown content to chunk.
            base_meta: Base metadata to include in each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        # Extract frontmatter and merge into metadata
        frontmatter, text = self._extract_frontmatter(text)
        if frontmatter:
            base_meta = {**base_meta, **frontmatter}

        doc_id = str(
            base_meta.get("doc_id") or base_meta.get("source_file") or "unknown"
        )

        # Split into sections
        sections = self._split_into_sections(text)

        # Optionally merge very small sections
        sections = self._merge_small_sections(sections)

        chunks: List[Chunk] = []
        chunk_index = 0

        for header, content, level in sections:
            if not content.strip():
                continue

            # Split large sections
            section_parts = self._split_large_section(header, content, level)

            for part_header, part_content in section_parts:
                if not part_content.strip():
                    continue

                chunk_meta = dict(base_meta)
                if part_header:
                    chunk_meta["section_header"] = part_header

                chunk_id = f"{doc_id}:{chunk_index}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        content=part_content,
                        metadata=chunk_meta,
                    )
                )
                chunk_index += 1

        return chunks


__all__ = ["MarkdownChunker"]
