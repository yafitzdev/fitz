# tests/unit/test_technical_doc_strategy.py
"""Tests for TechnicalDocIngestStrategy — section extraction from parsed documents."""

from __future__ import annotations

import pytest

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.engines.fitz_krag.ingestion.formats import DOCUMENT_EXTENSIONS
from fitz_sage.engines.fitz_krag.ingestion.strategies.technical_doc import (
    TechnicalDocIngestStrategy,
)


@pytest.fixture
def strategy():
    return TechnicalDocIngestStrategy()


def _make_doc(elements: list[DocumentElement], source: str = "test.pdf") -> ParsedDocument:
    return ParsedDocument(source=source, elements=elements)


class TestContentTypes:
    def test_includes_pdf(self, strategy):
        assert ".pdf" in strategy.content_types()

    def test_includes_docx(self, strategy):
        assert ".docx" in strategy.content_types()

    def test_includes_markdown(self, strategy):
        assert ".md" in strategy.content_types()

    def test_includes_txt(self, strategy):
        assert ".txt" in strategy.content_types()

    def test_matches_doc_extensions(self, strategy):
        assert strategy.content_types() == set(DOCUMENT_EXTENSIONS)


class TestNoHeadings:
    def test_empty_document_returns_empty(self, strategy):
        doc = _make_doc([])
        result = strategy.extract(doc, "test.pdf")
        assert result.sections == []

    def test_no_headings_single_section(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="First paragraph."),
                DocumentElement(type=ElementType.TEXT, content="Second paragraph."),
            ]
        )
        result = strategy.extract(doc, "docs/my_report.pdf")
        assert len(result.sections) == 1
        assert result.sections[0].title == "My Report"
        assert "First paragraph" in result.sections[0].content
        assert "Second paragraph" in result.sections[0].content
        assert result.sections[0].level == 1

    def test_no_headings_blank_content_returns_empty(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="   "),
            ]
        )
        result = strategy.extract(doc, "empty.pdf")
        assert result.sections == []

    def test_long_unheaded_document_is_split_without_losing_content(self, strategy):
        content = "X" * 13000
        doc = _make_doc(
            [
                DocumentElement(
                    type=ElementType.TEXT,
                    content=content,
                    page=7,
                )
            ]
        )

        result = strategy.extract(doc, "docs/reference_dump.txt")

        assert len(result.sections) == 3
        assert all(len(section.content) <= 6000 for section in result.sections)
        assert "".join(section.content for section in result.sections) == content
        assert [section.position for section in result.sections] == [0, 1, 2]
        assert [section.title for section in result.sections] == [
            "Reference Dump - Part 1",
            "Reference Dump - Part 2",
            "Reference Dump - Part 3",
        ]
        assert all(section.page_start == 7 for section in result.sections)
        assert all(section.page_end == 7 for section in result.sections)
        assert result.sections[0].metadata == {
            "document_title": "Reference Dump",
            "unheaded_part": 1,
            "unheaded_part_count": 3,
        }

    def test_unheaded_sections_preserve_element_page_ranges(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="A" * 4000, page=2),
                DocumentElement(type=ElementType.TEXT, content="B" * 4000, page=3),
            ]
        )

        result = strategy.extract(doc, "appendix.pdf")

        assert len(result.sections) == 2
        assert result.sections[0].content == "A" * 4000
        assert result.sections[0].page_start == 2
        assert result.sections[0].page_end == 2
        assert result.sections[1].content == "B" * 4000
        assert result.sections[1].page_start == 3
        assert result.sections[1].page_end == 3


class TestHeadingExtraction:
    def test_single_heading_with_content(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Introduction", level=1),
                DocumentElement(type=ElementType.TEXT, content="This is the intro."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert len(result.sections) == 1
        assert result.sections[0].title == "Introduction"
        assert result.sections[0].level == 1
        assert result.sections[0].content == "This is the intro."

    def test_multiple_headings(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Chapter 1", level=1),
                DocumentElement(type=ElementType.TEXT, content="Content of chapter 1."),
                DocumentElement(type=ElementType.HEADING, content="Chapter 2", level=1),
                DocumentElement(type=ElementType.TEXT, content="Content of chapter 2."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert len(result.sections) == 2
        assert result.sections[0].title == "Chapter 1"
        assert result.sections[1].title == "Chapter 2"
        assert result.sections[1].metadata["document_title"] == "Chapter 1"

    def test_nested_headings(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Main", level=1),
                DocumentElement(type=ElementType.TEXT, content="Main content."),
                DocumentElement(type=ElementType.HEADING, content="Sub", level=2),
                DocumentElement(type=ElementType.TEXT, content="Sub content."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert len(result.sections) == 2
        assert result.sections[0].level == 1
        assert result.sections[1].level == 2

    def test_heading_without_content_is_skipped(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Empty Section", level=1),
                DocumentElement(type=ElementType.HEADING, content="Full Section", level=1),
                DocumentElement(type=ElementType.TEXT, content="Some content."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert len(result.sections) == 1
        assert result.sections[0].title == "Full Section"


class TestPreamble:
    def test_preamble_before_first_heading(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="Preamble text before heading."),
                DocumentElement(type=ElementType.HEADING, content="Section 1", level=1),
                DocumentElement(type=ElementType.TEXT, content="Section 1 content."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert len(result.sections) == 2
        assert result.sections[0].title == "Introduction"
        assert "Preamble text" in result.sections[0].content
        assert result.sections[1].title == "Section 1"


class TestPageTracking:
    def test_page_numbers_tracked(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Results", level=1, page=5),
                DocumentElement(type=ElementType.TEXT, content="Results text.", page=5),
                DocumentElement(type=ElementType.TEXT, content="More results.", page=6),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert result.sections[0].page_start == 5
        assert result.sections[0].page_end == 6

    def test_no_pages_returns_none(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="Section", level=1),
                DocumentElement(type=ElementType.TEXT, content="Content."),
            ]
        )
        result = strategy.extract(doc, "test.md")
        assert result.sections[0].page_start is None
        assert result.sections[0].page_end is None


class TestParentAssignment:
    def test_h2_gets_h1_parent(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="H1", level=1),
                DocumentElement(type=ElementType.TEXT, content="H1 content."),
                DocumentElement(type=ElementType.HEADING, content="H2", level=2),
                DocumentElement(type=ElementType.TEXT, content="H2 content."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        # H1 has no parent
        assert result.sections[0].parent_id is None
        # H2 should have parent referencing index 0
        assert result.sections[1].parent_id == "_parent_0"

    def test_h3_gets_h2_parent(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="H1", level=1),
                DocumentElement(type=ElementType.TEXT, content="H1 content."),
                DocumentElement(type=ElementType.HEADING, content="H2", level=2),
                DocumentElement(type=ElementType.TEXT, content="H2 content."),
                DocumentElement(type=ElementType.HEADING, content="H3", level=3),
                DocumentElement(type=ElementType.TEXT, content="H3 content."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        assert result.sections[2].parent_id == "_parent_1"


class TestPositions:
    def test_positions_are_sequential(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.HEADING, content="A", level=1),
                DocumentElement(type=ElementType.TEXT, content="A content."),
                DocumentElement(type=ElementType.HEADING, content="B", level=1),
                DocumentElement(type=ElementType.TEXT, content="B content."),
                DocumentElement(type=ElementType.HEADING, content="C", level=1),
                DocumentElement(type=ElementType.TEXT, content="C content."),
            ]
        )
        result = strategy.extract(doc, "test.pdf")
        positions = [s.position for s in result.sections]
        assert positions == [0, 1, 2]


class TestTitleFromPath:
    def test_underscores_converted(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="Content."),
            ]
        )
        result = strategy.extract(doc, "my_test_doc.pdf")
        assert result.sections[0].title == "My Test Doc"

    def test_hyphens_converted(self, strategy):
        doc = _make_doc(
            [
                DocumentElement(type=ElementType.TEXT, content="Content."),
            ]
        )
        result = strategy.extract(doc, "path/to/api-reference.md")
        assert result.sections[0].title == "Api Reference"
