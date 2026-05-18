# tests/unit/test_cpu_pdf_parser.py
"""Unit tests for the CPU-first PDF parser (roadmap v-0-13-05).

The heading heuristics — numbering, font-size clustering, all-caps, keyword
labels — are tested directly on reconstructed lines (deterministic, no PDF).
One integration test parses a real fixture PDF end to end.
"""

from __future__ import annotations

from pathlib import Path

from fitz_sage.core.document import ElementType
from fitz_sage.ingestion.parser.plugins.cpu_pdf import CpuPdfParser, _Line
from fitz_sage.ingestion.source.base import SourceFile

_SAMPLE_PDF = Path(__file__).resolve().parent.parent / "e2e_krag" / "fixtures_parser" / "sample.pdf"


def _line(text: str, font_size: float = 10.0, page: int = 1) -> _Line:
    return _Line(text=text, font_size=font_size, page=page)


# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------


class TestHeadingClassification:
    """_heading_level — the heuristic that drives section structure."""

    def test_numbered_heading_level_from_depth(self):
        """A numbered heading's level is its numbering depth."""
        assert CpuPdfParser._heading_level(_line("1 Introduction"), {}) == 1
        assert CpuPdfParser._heading_level(_line("2.3 Retrieval Sources"), {}) == 2
        assert CpuPdfParser._heading_level(_line("4.1.2 Adaptive Retrieval"), {}) == 3

    def test_numbered_list_item_is_not_heading(self):
        """A long, body-font numbered line is a list item, not a heading."""
        line = _line("3. Configure the database connection and run every pending migration")
        assert CpuPdfParser._heading_level(line, {}) is None

    def test_numbered_sentence_is_not_heading(self):
        """A numbered line ending in a period reads as a sentence."""
        assert CpuPdfParser._heading_level(_line("1. Do this first."), {}) is None

    def test_toc_entry_rejected(self):
        """A numbered line with a trailing page number is a table-of-contents row."""
        assert CpuPdfParser._heading_level(_line("6 AI RMF Profiles 33"), {}) is None

    def test_dotted_leader_rejected(self):
        """Dotted leaders mark a table-of-contents row, never a heading."""
        assert CpuPdfParser._heading_level(_line("Introduction .......... 4"), {}) is None

    def test_page_header_rejected(self):
        """Running page headers/footers are never headings."""
        assert CpuPdfParser._heading_level(_line("Page 5 of 126"), {}) is None

    def test_font_size_cluster_heading(self):
        """A line in an above-body font size is a heading at that size's level."""
        assert CpuPdfParser._heading_level(_line("Section Title", 16.0), {16.0: 1, 13.0: 2}) == 1
        assert CpuPdfParser._heading_level(_line("Subsection", 13.0), {16.0: 1, 13.0: 2}) == 2

    def test_all_caps_short_line_is_heading(self):
        """A short all-caps line is a heading even without font or numbering cues."""
        assert CpuPdfParser._heading_level(_line("OVERVIEW OF RAG"), {}) == 2

    def test_keyword_heading(self):
        """'Chapter 3' / 'Part II' style labels are headings."""
        assert CpuPdfParser._heading_level(_line("Chapter 3"), {}) == 1
        assert CpuPdfParser._heading_level(_line("Part II"), {}) == 1

    def test_keyword_sentence_is_not_heading(self):
        """A sentence that merely starts with 'Part 2' is not a heading."""
        line = _line("Part 2 comprises the core of the framework and its profiles")
        assert CpuPdfParser._heading_level(line, {}) is None

    def test_plain_body_text_is_not_heading(self):
        """Ordinary body text is not a heading."""
        line = _line("This is an ordinary sentence of body text in the document.")
        assert CpuPdfParser._heading_level(line, {}) is None

    def test_overlong_line_is_not_heading(self):
        """A line past the length limit is body text regardless of other cues."""
        assert CpuPdfParser._heading_level(_line("WORD " * 40), {}) is None


# ---------------------------------------------------------------------------
# Font-size analysis
# ---------------------------------------------------------------------------


class TestFontAnalysis:
    def test_body_font_size_is_the_most_common(self):
        """Body font size = the size carrying the most text."""
        lines = [_line("aaaaaaaa", 10.0), _line("bbbbbbbb", 10.0), _line("c", 18.0)]
        assert CpuPdfParser._body_font_size(lines) == 10.0

    def test_heading_size_levels_rank_by_size(self):
        """Above-body sizes map to heading levels, largest first."""
        lines = [_line("body", 10.0), _line("big", 20.0), _line("mid", 14.0)]
        assert CpuPdfParser._heading_size_levels(lines, body_size=10.0) == {20.0: 1, 14.0: 2}

    def test_near_body_sizes_are_not_headings(self):
        """A size barely above body (< +1.0) is not treated as a heading size."""
        lines = [_line("body", 10.0), _line("almost", 10.5)]
        assert CpuPdfParser._heading_size_levels(lines, body_size=10.0) == {}


# ---------------------------------------------------------------------------
# Element building
# ---------------------------------------------------------------------------


class TestBuildElements:
    def test_repeated_label_is_suppressed(self):
        """A 'heading' that recurs many times is a callout label, not a section."""
        parser = CpuPdfParser()
        lines = []
        for i in range(10):
            lines.append(_line("CAUTION", page=i + 1))
            lines.append(_line(f"Body paragraph number {i} with some detail.", page=i + 1))
        elements = parser._build_elements(lines)
        headings = [e for e in elements if e.type == ElementType.HEADING]
        assert not any("CAUTION" in h.content for h in headings)

    def test_headings_and_body_alternate(self):
        """Headings emit HEADING elements; runs of body lines coalesce into TEXT."""
        parser = CpuPdfParser()
        lines = [
            _line("1 Introduction"),
            _line("First body line."),
            _line("Second body line."),
            _line("2 Methods"),
            _line("Methods body line."),
        ]
        elements = parser._build_elements(lines)
        kinds = [(e.type, e.content) for e in elements]
        assert kinds[0] == (ElementType.HEADING, "1 Introduction")
        assert kinds[1][0] == ElementType.TEXT
        assert "First body line." in kinds[1][1] and "Second body line." in kinds[1][1]
        assert kinds[2] == (ElementType.HEADING, "2 Methods")


# ---------------------------------------------------------------------------
# Integration — a real PDF
# ---------------------------------------------------------------------------


class TestParseRealPdf:
    def test_parse_sample_pdf(self):
        """Parsing a real digital PDF yields page count, headings, and body text."""
        doc = CpuPdfParser().parse(SourceFile(uri=_SAMPLE_PDF.as_uri(), local_path=_SAMPLE_PDF))

        assert doc.metadata["page_count"] == 2

        headings = [e for e in doc.elements if e.type == ElementType.HEADING]
        body = [e for e in doc.elements if e.type == ElementType.TEXT]
        assert headings, "expected at least one heading"
        assert body, "expected at least one body element"

        titles = " | ".join(h.content for h in headings)
        assert "Nexus Robotics" in titles
        assert "Company Overview" in titles
