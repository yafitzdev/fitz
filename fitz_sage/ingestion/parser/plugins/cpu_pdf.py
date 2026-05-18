# fitz_sage/ingestion/parser/plugins/cpu_pdf.py
"""
CPU-first PDF parser — pypdfium2 text layer, zero models, server-free.

The v1 path of the CPU-first parser (roadmap v-0-13-05): for digital text
PDFs — the common case — the text and its font metadata are already in the
file. ``pypdfium2`` reads them and this parser reconstructs heading hierarchy
from font-size clustering plus numbering patterns. No layout transformer, no
OCR, no torch — so it cannot exhaust memory the way the docling pipeline does
(it rasterizes every page) and it runs in seconds, not minutes.

Scanned / image-only PDFs have no text layer to read; this parser returns an
empty document for them. Use ``parser: docling`` for OCR on those.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Set

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.ingestion.parser.base import ParseError
from fitz_sage.ingestion.source.base import SourceFile

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# Numbered section headings: "1 ", "1. ", "2.3 ", "4.1.2 " — the leading
# number group's depth (dot count + 1) gives the heading level.
_NUMBERED_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){0,4})\.?\s+\S")
# Keyword headings: "Chapter 3", "Appendix A", "Part II".
_KEYWORD_RE = re.compile(r"^(chapter|section|part|appendix|article)\s+[\divxlcDIVXLC]+\b", re.I)
# Running page headers/footers — never a heading.
_PAGE_HEADER_RE = re.compile(r"page\s+\d+\s+of\s+\d+", re.I)

# A line longer than this — or with more words than this — is body text.
_MAX_HEADING_LEN = 100
_MAX_HEADING_WORDS = 14
_MAX_LEVEL = 6
# A "heading" text that recurs at least this many times is a repeated label
# (callout boxes like TIP / CAUTION, form-field labels), not a section heading.
_REPEAT_SUPPRESS = 8


@dataclass
class _Line:
    """One reconstructed line of text with its dominant font size."""

    text: str
    font_size: float  # 0.0 when pdfium could not report it
    page: int


@dataclass
class CpuPdfParser(BaseParser):
    """Zero-model PDF parser: reads the text layer, infers headings from fonts."""

    plugin_name: str = field(default="cpu_pdf")
    supported_extensions: Set[str] = field(default_factory=lambda: {".pdf"})

    def parse(self, file: SourceFile) -> ParsedDocument:
        try:
            import pypdfium2 as pdfium
        except ImportError as e:  # pragma: no cover - dependency is in base install
            raise ImportError(
                "pypdfium2 is required for the cpu_pdf parser. "
                "Install it with: pip install pypdfium2"
            ) from e

        data = self._read_file_bytes(file)
        try:
            pdf = pdfium.PdfDocument(data)
        except Exception as e:
            raise ParseError(f"Failed to open PDF: {e}", source=file.uri, cause=e) from e

        try:
            page_count = len(pdf)
            lines: list[_Line] = []
            for page_idx in range(page_count):
                lines.extend(self._extract_lines(pdf[page_idx], page_idx + 1))
        finally:
            pdf.close()

        self._drop_repeated_furniture(lines, page_count)
        elements = self._build_elements(lines)

        if not elements:
            logger.warning(
                "cpu_pdf extracted no text from %s — it may be a scanned/image PDF. "
                "Use 'parser: docling' for OCR on scanned documents.",
                file.uri,
            )

        return ParsedDocument(
            source=file.uri,
            elements=elements,
            metadata=self._build_metadata(file, page_count=page_count),
        )

    # ------------------------------------------------------------------
    # Line extraction
    # ------------------------------------------------------------------

    def _extract_lines(self, page: Any, page_num: int) -> list[_Line]:
        """Pull text rects from one page and merge them into reconstructed lines."""
        import pypdfium2.raw as pdfium_c

        textpage = page.get_textpage()
        try:
            rects: list[tuple[float, float, float, float, str, float]] = []
            for i in range(textpage.count_rects()):
                left, bottom, right, top = textpage.get_rect(i)
                text = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)
                if not text or not text.strip():
                    continue
                font_size = self._rect_font_size(textpage, pdfium_c, left, bottom, right, top)
                rects.append((left, bottom, right, top, text, font_size))
        finally:
            textpage.close()

        # pdfium emits rects in reading order but splits a visual line wherever
        # the font changes (superscripts, emphasis). Merge rects back into lines
        # by vertical proximity.
        lines: list[_Line] = []
        bucket: list[tuple[float, float, float, float, str, float]] = []

        def flush() -> None:
            if not bucket:
                return
            ordered = sorted(bucket, key=lambda r: r[0])
            # Join fragments: no space when rects touch (drop caps, superscripts),
            # a space when there is a real word gap.
            parts: list[str] = []
            prev_right: float | None = None
            for left, _b, right, _t, frag, _fs in ordered:
                frag = frag.strip()
                if not frag:
                    continue
                if prev_right is not None and left - prev_right > 1.0:
                    parts.append(" ")
                parts.append(frag)
                prev_right = right
            text = re.sub(r"\s+", " ", "".join(parts)).strip()
            if text:
                weights: Counter = Counter()
                for r in ordered:
                    if r[5] > 1.5:
                        weights[r[5]] += max(len(r[4].strip()), 1)
                font_size = weights.most_common(1)[0][0] if weights else 0.0
                lines.append(_Line(text=text, font_size=font_size, page=page_num))
            bucket.clear()

        for rect in rects:
            _, bottom, _, top, _, _ = rect
            y_center = (bottom + top) / 2
            if bucket:
                cur_top = max(r[3] for r in bucket)
                cur_bottom = min(r[1] for r in bucket)
                ref_height = max(cur_top - cur_bottom, 1.0)
                if abs(y_center - (cur_top + cur_bottom) / 2) > ref_height * 0.7:
                    flush()
            bucket.append(rect)
        flush()
        return lines

    @staticmethod
    def _rect_font_size(
        textpage: Any, pdfium_c: Any, left: float, bottom: float, right: float, top: float
    ) -> float:
        """Sample the font size of a text rect via pdfium's per-char font size.

        Returns 0.0 when pdfium reports nothing — whitespace-only chars report a
        size of 1.0, so anything at or below 1.5 is treated as 'unknown'.
        """
        width, height = right - left, top - bottom
        y = (bottom + top) / 2
        sizes: list[float] = []
        for frac in (0.2, 0.5, 0.8):
            idx = textpage.get_index(
                left + width * frac, y, max(width * 0.2, 1.0), max(height * 0.5, 1.0)
            )
            if idx is None or idx < 0:
                continue
            try:
                size = pdfium_c.FPDFText_GetFontSize(textpage.raw, idx)
            except Exception:  # pragma: no cover - defensive
                continue
            if size and size > 1.5:
                sizes.append(float(size))
        if not sizes:
            return 0.0
        sizes.sort()
        return round(sizes[len(sizes) // 2] * 2) / 2  # median, snapped to 0.5

    @staticmethod
    def _drop_repeated_furniture(lines: list[_Line], page_count: int) -> None:
        """Remove running headers / footers / page numbers — lines that repeat
        across most pages — so they do not pollute the heading structure."""
        if page_count < 4:
            return

        def normalize(text: str) -> str:
            return re.sub(r"\d+", "#", text.strip().lower())

        pages_seen: defaultdict[str, set[int]] = defaultdict(set)
        for line in lines:
            pages_seen[normalize(line.text)].add(line.page)

        threshold = max(3, int(page_count * 0.5))
        furniture = {key for key, pages in pages_seen.items() if len(pages) >= threshold}
        if furniture:
            lines[:] = [line for line in lines if normalize(line.text) not in furniture]

    # ------------------------------------------------------------------
    # Heading inference
    # ------------------------------------------------------------------

    def _build_elements(self, lines: list[_Line]) -> list[DocumentElement]:
        """Classify each line as heading or body and emit document elements."""
        if not lines:
            return []

        body_size = self._body_font_size(lines)
        size_levels = self._heading_size_levels(lines, body_size)

        # Classify every line, then suppress "headings" whose text recurs many
        # times — repeated callout labels are not section headings.
        levels = [self._heading_level(line, size_levels) for line in lines]
        heading_counts = Counter(
            line.text for line, level in zip(lines, levels) if level is not None
        )
        repeated = {text for text, count in heading_counts.items() if count >= _REPEAT_SUPPRESS}

        elements: list[DocumentElement] = []
        body_buffer: list[str] = []
        body_page: int | None = None

        def flush_body() -> None:
            nonlocal body_page
            if body_buffer:
                text = "\n".join(body_buffer).strip()
                if text:
                    elements.append(
                        DocumentElement(type=ElementType.TEXT, content=text, page=body_page)
                    )
                body_buffer.clear()
                body_page = None

        for line, level in zip(lines, levels):
            if level is not None and line.text not in repeated:
                flush_body()
                elements.append(
                    DocumentElement(
                        type=ElementType.HEADING,
                        content=line.text,
                        level=level,
                        page=line.page,
                    )
                )
            else:
                if body_page is None:
                    body_page = line.page
                body_buffer.append(line.text)
        flush_body()
        return elements

    @staticmethod
    def _body_font_size(lines: list[_Line]) -> float:
        """The font size carrying the most text — the document's body size."""
        weights: Counter = Counter()
        for line in lines:
            if line.font_size > 1.5:
                weights[line.font_size] += len(line.text)
        return weights.most_common(1)[0][0] if weights else 0.0

    @staticmethod
    def _heading_size_levels(lines: list[_Line], body_size: float) -> dict[float, int]:
        """Map each above-body font size to a heading level (largest = level 1)."""
        if body_size <= 0:
            return {}
        bigger = sorted(
            {line.font_size for line in lines if line.font_size >= body_size + 1.0},
            reverse=True,
        )
        return {size: min(rank + 1, _MAX_LEVEL) for rank, size in enumerate(bigger)}

    @staticmethod
    def _heading_level(line: _Line, size_levels: dict[float, int]) -> int | None:
        """Return a heading level for the line, or None if it is body text."""
        text = line.text.strip()
        if not text or len(text) > _MAX_HEADING_LEN:
            return None
        if len(text.split()) > _MAX_HEADING_WORDS:
            return None
        # Table-of-contents lines (dotted leaders) and running page headers
        # are never headings.
        if re.search(r"\.{3,}", text) or _PAGE_HEADER_RE.search(text):
            return None

        is_big = line.font_size in size_levels

        # 1. Numbered heading — level from the numbering depth. Reject sentences
        #    (trailing period), TOC entries (trailing page number), and long
        #    body-font list items.
        numbered = _NUMBERED_RE.match(text)
        if numbered:
            rest = text[numbered.end(1) :].strip()
            looks_like_toc = bool(re.search(r"\s\d{1,4}$", rest))
            if not text.endswith(".") and not looks_like_toc and (is_big or len(text.split()) <= 9):
                return min(numbered.group(1).count(".") + 1, _MAX_LEVEL)

        # 2. Keyword heading — "Chapter 3", "Appendix A" — only when the line
        #    is the label itself, not a sentence that starts with the word and
        #    not a TOC entry (trailing page number after the keyword's number).
        keyword = _KEYWORD_RE.match(text)
        if keyword and len(text) <= 70:
            after = text[keyword.end() :]
            if not re.search(r"\s\d{1,4}$", after):
                after = after.lstrip()
                if not after or after[0] in ":.—-" or after[0].isupper():
                    return 1

        # 3. Font-size cluster — a size meaningfully larger than body text.
        if is_big:
            return size_levels[line.font_size]

        # 4. All-caps short line — a common heading style in technical docs.
        if (
            len(text) <= 80
            and text == text.upper()
            and any(ch.isalpha() for ch in text)
            and not text.endswith((".", ",", ";", ":"))
            and len(text.split()) <= 12
        ):
            return 2

        return None


__all__ = ["CpuPdfParser"]
