# fitz_ai/ingest/chunking/plugins/pdf_sections.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass
class PdfSectionChunker:
    """
    Section-based PDF chunker.

    Detects sections in PDF text by identifying:
    - Lines with all caps (common for headers)
    - Lines with title case followed by newlines
    - Numbered sections (1. 2. 3. or 1.1, 1.2, etc.)
    - Common section keywords (Introduction, Abstract, Conclusion, etc.)

    Chunks are created for each detected section, preserving document structure.

    Args:
        plugin_name: Plugin identifier
        min_section_chars: Minimum characters for a valid section (default: 50)
        max_section_chars: Maximum characters per section before splitting (default: 3000)
        preserve_short_sections: If True, keep sections shorter than min_section_chars (default: True)
    """

    plugin_name: str = "pdf_sections"
    min_section_chars: int = 50
    max_section_chars: int = 3000
    preserve_short_sections: bool = True

    # Section header patterns
    _SECTION_PATTERNS = [
        # Numbered sections: "1.", "1.1", "1.1.1", etc.
        re.compile(r"^\d+(\.\d+)*\.\s+[A-Z]"),
        # Roman numerals: "I.", "II.", etc.
        re.compile(r"^[IVX]+\.\s+[A-Z]"),
        # Letter sections: "A.", "B.", etc.
        re.compile(r"^[A-Z]\.\s+[A-Z]"),
    ]

    _SECTION_KEYWORDS = [
        "abstract",
        "introduction",
        "background",
        "methodology",
        "methods",
        "results",
        "discussion",
        "conclusion",
        "references",
        "bibliography",
        "appendix",
        "acknowledgments",
        "summary",
    ]

    def _is_section_header(self, line: str) -> bool:
        """
        Determine if a line is likely a section header.

        Detection heuristics:
        1. All caps line (3-50 chars)
        2. Starts with numbered pattern
        3. Contains section keyword and is title case
        4. Short line (<80 chars) that's mostly uppercase
        """
        stripped = line.strip()

        if not stripped or len(stripped) < 3:
            return False

        # Check for all caps (common in PDF headers)
        if stripped.isupper() and 3 <= len(stripped) <= 50:
            return True

        # Check numbered patterns
        for pattern in self._SECTION_PATTERNS:
            if pattern.match(stripped):
                return True

        # Check for section keywords in title case
        lower = stripped.lower()
        for keyword in self._SECTION_KEYWORDS:
            if keyword in lower and stripped[0].isupper():
                # Title case check: first letter uppercase, rest mostly lowercase
                words = stripped.split()
                if len(words) <= 4 and any(w[0].isupper() for w in words if w):
                    return True

        # Check for short lines with high uppercase ratio (potential headers)
        if len(stripped) < 80:
            upper_count = sum(1 for c in stripped if c.isupper())
            alpha_count = sum(1 for c in stripped if c.isalpha())
            if alpha_count > 0 and upper_count / alpha_count > 0.6:
                return True

        return False

    def _split_into_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Split text into sections based on detected headers.

        Returns:
            List of (header, content) tuples
        """
        lines = text.split("\n")
        sections: List[tuple[str, str]] = []

        current_header = "Preamble"
        current_content: List[str] = []

        for line in lines:
            if self._is_section_header(line):
                # Save previous section
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        sections.append((current_header, content))

                # Start new section
                current_header = line.strip()
                current_content = []
            else:
                current_content.append(line)

        # Save final section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections.append((current_header, content))

        return sections

    def _split_large_section(
        self, header: str, content: str, max_chars: int
    ) -> List[tuple[str, str]]:
        """
        Split a large section into smaller chunks while preserving context.

        Splits on paragraph boundaries when possible.
        """
        if len(content) <= max_chars:
            return [(header, content)]

        chunks: List[tuple[str, str]] = []
        paragraphs = content.split("\n\n")

        current_chunk: List[str] = []
        current_size = 0
        part_num = 1

        for para in paragraphs:
            para_size = len(para)

            # If single paragraph exceeds max, split it
            if para_size > max_chars:
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk).strip()
                    chunk_header = (
                        f"{header} (Part {part_num})" if part_num > 1 else header
                    )
                    chunks.append((chunk_header, chunk_content))
                    part_num += 1
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                temp_chunk: List[str] = []
                temp_size = 0

                for sent in sentences:
                    if temp_size + len(sent) > max_chars and temp_chunk:
                        chunk_content = " ".join(temp_chunk).strip()
                        chunk_header = f"{header} (Part {part_num})"
                        chunks.append((chunk_header, chunk_content))
                        part_num += 1
                        temp_chunk = [sent]
                        temp_size = len(sent)
                    else:
                        temp_chunk.append(sent)
                        temp_size += len(sent)

                if temp_chunk:
                    chunk_content = " ".join(temp_chunk).strip()
                    chunk_header = f"{header} (Part {part_num})"
                    chunks.append((chunk_header, chunk_content))
                    part_num += 1

            # Regular paragraph handling
            elif current_size + para_size > max_chars and current_chunk:
                chunk_content = "\n\n".join(current_chunk).strip()
                chunk_header = f"{header} (Part {part_num})" if part_num > 1 else header
                chunks.append((chunk_header, chunk_content))
                part_num += 1
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Save final chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk).strip()
            chunk_header = f"{header} (Part {part_num})" if part_num > 1 else header
            chunks.append((chunk_header, chunk_content))

        return chunks

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk PDF text by detected sections.

        Args:
            text: Raw PDF text content
            base_meta: Base metadata including doc_id, source_file, etc.

        Returns:
            List of Chunk objects, one per section (or section part)
        """
        doc_id = str(
            base_meta.get("doc_id") or base_meta.get("source_file") or "unknown"
        )

        # Split into sections
        sections = self._split_into_sections(text)

        if not sections:
            # Fallback: treat entire document as one section
            sections = [("Document", text)]

        chunks: List[Chunk] = []
        chunk_index = 0

        for header, content in sections:
            # Skip very short sections unless preserve_short_sections is True
            if (
                not self.preserve_short_sections
                and len(content) < self.min_section_chars
            ):
                continue

            # Split large sections if needed
            section_parts = self._split_large_section(
                header, content, self.max_section_chars
            )

            for part_header, part_content in section_parts:
                if not part_content.strip():
                    continue

                # Create metadata with section information
                chunk_meta = dict(base_meta)
                chunk_meta["section_header"] = part_header
                chunk_meta["original_header"] = header

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
