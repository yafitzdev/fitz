# fitz_sage/engines/fitz_krag/ingestion/strategies/technical_doc.py
"""
Technical document ingestion strategy.

Extracts sections from parsed documents (PDFs, DOCX, Markdown) using
the existing Docling/PlainText parser's structural elements. Sections
are grouped by HEADING elements into a hierarchical tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.engines.fitz_krag.ingestion.formats import DOCUMENT_EXTENSIONS

logger = logging.getLogger(__name__)

_UNHEADED_SECTION_MAX_CHARS = 6000


@dataclass
class SectionEntry:
    """A document section extracted from a parsed document."""

    title: str
    level: int
    content: str
    page_start: int | None = None
    page_end: int | None = None
    parent_id: str | None = None
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocIngestResult:
    """Result of extracting sections from a document."""

    sections: list[SectionEntry] = field(default_factory=list)


class TechnicalDocIngestStrategy:
    """Extracts sections from technical documents using parsed document elements."""

    def content_types(self) -> set[str]:
        return set(DOCUMENT_EXTENSIONS)

    def extract(self, parsed_doc: ParsedDocument, file_path: str) -> DocIngestResult:
        """
        Extract sections from a parsed document.

        Groups elements between HEADING markers into sections.
        Builds hierarchical tree from heading levels (H1 > H2 > H3).
        """
        elements = parsed_doc.elements
        if not elements:
            return DocIngestResult()

        # Check if document has any headings
        headings = [el for el in elements if el.type == ElementType.HEADING]

        if not headings:
            return DocIngestResult(sections=_build_unheaded_sections(elements, file_path))

        # Build sections from heading structure
        sections = self._build_sections(elements)
        document_title = headings[0].content.strip()
        if document_title:
            for section in sections:
                section.metadata.setdefault("document_title", document_title)
        return DocIngestResult(sections=sections)

    def _build_sections(self, elements: list[DocumentElement]) -> list[SectionEntry]:
        """Build section list from document elements."""
        sections: list[SectionEntry] = []
        current_title: str | None = None
        current_level: int = 1
        current_content_parts: list[str] = []
        current_page_start: int | None = None
        current_page_end: int | None = None
        position = 0

        # Collect pre-heading content
        preamble_parts: list[str] = []

        for el in elements:
            if el.type == ElementType.HEADING:
                # Save previous section
                if current_title is not None:
                    content = "\n\n".join(current_content_parts).strip()
                    if content:
                        sections.append(
                            SectionEntry(
                                title=current_title,
                                level=current_level,
                                content=content,
                                page_start=current_page_start,
                                page_end=current_page_end,
                                position=position,
                            )
                        )
                        position += 1
                elif preamble_parts:
                    # Content before first heading
                    content = "\n\n".join(preamble_parts).strip()
                    if content:
                        sections.append(
                            SectionEntry(
                                title="Introduction",
                                level=1,
                                content=content,
                                page_start=_first_page_of(preamble_parts, elements),
                                position=position,
                            )
                        )
                        position += 1

                # Start new section
                current_title = el.content.strip()
                current_level = el.level or 1
                current_content_parts = []
                current_page_start = el.page
                current_page_end = el.page
            else:
                if el.content and el.content.strip():
                    if current_title is None:
                        preamble_parts.append(el.content)
                    else:
                        current_content_parts.append(el.content)
                        if el.page is not None:
                            current_page_end = el.page

        # Save last section
        if current_title is not None:
            content = "\n\n".join(current_content_parts).strip()
            if content:
                sections.append(
                    SectionEntry(
                        title=current_title,
                        level=current_level,
                        content=content,
                        page_start=current_page_start,
                        page_end=current_page_end,
                        position=position,
                    )
                )

        # Build parent-child hierarchy
        self._assign_parents(sections)

        return sections

    def _assign_parents(self, sections: list[SectionEntry]) -> None:
        """Assign parent IDs based on heading levels."""
        # Use a stack of (level, section_index) to track hierarchy
        parent_stack: list[tuple[int, int]] = []

        for i, section in enumerate(sections):
            # Pop stack until we find a parent with lower level
            while parent_stack and parent_stack[-1][0] >= section.level:
                parent_stack.pop()

            if parent_stack:
                # Parent is the top of the stack
                parent_idx = parent_stack[-1][1]
                section.parent_id = f"_parent_{parent_idx}"
                # This is a placeholder — actual IDs assigned during storage

            parent_stack.append((section.level, i))


def _title_from_path(file_path: str) -> str:
    """Generate a title from file path when no headings exist."""
    import os

    name = os.path.basename(file_path)
    name = os.path.splitext(name)[0]
    return name.replace("_", " ").replace("-", " ").title()


def _build_unheaded_sections(
    elements: list[DocumentElement],
    file_path: str,
) -> list[SectionEntry]:
    chunks = _chunk_unheaded_elements(elements)
    if not chunks:
        return []

    document_title = _title_from_path(file_path)
    chunk_count = len(chunks)
    sections: list[SectionEntry] = []
    for position, (content, page_start, page_end) in enumerate(chunks):
        metadata: dict[str, Any] = {}
        title = document_title
        if chunk_count > 1:
            title = f"{document_title} - Part {position + 1}"
            metadata = {
                "document_title": document_title,
                "unheaded_part": position + 1,
                "unheaded_part_count": chunk_count,
            }
        sections.append(
            SectionEntry(
                title=title,
                level=1,
                content=content,
                page_start=page_start,
                page_end=page_end,
                position=position,
                metadata=metadata,
            )
        )
    return sections


def _chunk_unheaded_elements(
    elements: list[DocumentElement],
) -> list[tuple[str, int | None, int | None]]:
    chunks: list[tuple[str, int | None, int | None]] = []
    parts: list[str] = []
    content_length = 0
    page_start: int | None = None
    page_end: int | None = None

    def flush() -> None:
        nonlocal content_length, page_start, page_end
        if parts:
            chunks.append(("\n\n".join(parts), page_start, page_end))
        parts.clear()
        content_length = 0
        page_start = None
        page_end = None

    for element in elements:
        for fragment in _split_unheaded_text(element.content):
            separator_length = 2 if parts else 0
            if parts and (
                content_length + separator_length + len(fragment)
                > _UNHEADED_SECTION_MAX_CHARS
            ):
                flush()
                separator_length = 0

            if not parts:
                page_start = element.page
            elif page_start is None and element.page is not None:
                page_start = element.page
            parts.append(fragment)
            content_length += separator_length + len(fragment)
            if element.page is not None:
                page_end = element.page

    flush()
    return chunks


def _split_unheaded_text(value: str) -> list[str]:
    remaining = value.strip()
    if not remaining:
        return []

    fragments: list[str] = []
    while len(remaining) > _UNHEADED_SECTION_MAX_CHARS:
        boundary = _preferred_split_boundary(remaining)
        fragments.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        fragments.append(remaining)
    return fragments


def _preferred_split_boundary(value: str) -> int:
    lower_bound = _UNHEADED_SECTION_MAX_CHARS // 2
    upper_bound = _UNHEADED_SECTION_MAX_CHARS + 1
    for separator in ("\n\n", "\n", " "):
        boundary = value.rfind(separator, lower_bound, upper_bound)
        if boundary >= lower_bound:
            return boundary
    return _UNHEADED_SECTION_MAX_CHARS


def _first_page(elements: list[DocumentElement]) -> int | None:
    """Get the first page number from elements."""
    for el in elements:
        if el.page is not None:
            return el.page
    return None


def _first_page_of(parts: list[str], elements: list[DocumentElement]) -> int | None:
    """Get first page for preamble content."""
    return _first_page(elements)
