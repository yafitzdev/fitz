# fitz_sage/ingestion/parser/plugins/glm_ocr.py
"""
Hybrid PDF parser — pdfplumber fast path + GLM-OCR fallback for scanned pages.

Per-page routing:
  1. pdfplumber extracts text with font sizes (instant, structured)
  2. Pages with enough text → heading detection via font size
  3. Scanned pages (no text) → rendered to image → GLM-OCR via ollama

pdfplumber gives us font sizes (heading hierarchy), table detection,
and image detection — all without ML. GLM-OCR handles truly scanned content.

Requires for OCR fallback: ollama pull glm-ocr (2.2GB, MIT licensed)
"""

from __future__ import annotations

import base64
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Set

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.ingestion.source.base import SourceFile

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

GLM_OCR_EXTENSIONS: Set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# Minimum text chars to consider a page "text-extractable"
_MIN_TEXT_CHARS = 50

# Markdown patterns for GLM-OCR output parsing
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_CODE_FENCE_RE = re.compile(r"^```")

# Section heading patterns (for PDFs without font-size differentiation)
# Matches: "I. INTRODUCTION", "A. Naive RAG", "1.2.3 Risk", "IV. GENERATION"
_SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:[IVXLC]+\.\s+[A-Z])"  # Roman numeral: "I. INTRO", "IV. GEN"
    r"|(?:[A-H]\.\s+[A-Z])"  # Letter: "A. Naive", "B. Advanced"
    r"|(?:\d+(?:\.\d+)*\s+[A-Z])"  # Numbered: "1. Framing", "3.2 Safe"
    r")"
)


# CamelCase split: insert space before uppercase letter preceded by lowercase
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")

# Noise heading patterns: bullet points, single chars, form metadata
_NOISE_HEADING_RE = re.compile(
    r"^(?:"
    r"[•·\-–—]\s"  # Bullet points
    r"|Page \d+"  # Page numbers
    r"|\d{1,2}:\d{2}\s"  # Timestamps (15:15)
    r"|AH XSL"  # PDF tool metadata
    r"|Fileid:"  # Form file IDs
    r"|[A-Z]{2,}\s*$"  # Short all-caps (form codes)
    r")",
    re.IGNORECASE,
)


def _fix_nospace_title(title: str) -> str:
    """Insert spaces into camelCase titles: 'RiskMeasurement' → 'Risk Measurement'."""
    if " " in title:
        return title
    if not any(c.isupper() for c in title[1:]):
        return title
    return _CAMEL_RE.sub(r"\1 \2", title)


def _is_noise_heading(title: str) -> bool:
    """Check if a heading is noise that should be demoted to TEXT."""
    if len(title) <= 2:
        return True
    if _NOISE_HEADING_RE.match(title):
        return True
    # Pure numbers or very short fragments
    if title.strip().replace(".", "").replace(",", "").isdigit():
        return True
    return False


@dataclass
class GlmOcrParser(BaseParser):
    """Hybrid PDF parser: pdfplumber for text pages, GLM-OCR for scanned pages."""

    plugin_name: str = field(default="glm_ocr")
    supported_extensions: Set[str] = field(default_factory=lambda: GLM_OCR_EXTENSIONS)
    model: str = field(default="glm-ocr")
    base_url: str = field(default="http://localhost:11434")
    dpi: int = field(default=150)

    def parse(self, file: SourceFile) -> ParsedDocument:
        if file.extension == ".pdf":
            return self._parse_pdf(file)
        return self._parse_image(file)

    def _parse_pdf(self, file: SourceFile) -> ParsedDocument:
        """Parse PDF with per-page routing: pdfplumber fast path + GLM-OCR fallback."""
        import pdfplumber

        pdf = pdfplumber.open(file.local_path)
        page_count = len(pdf.pages)

        elements: List[DocumentElement] = []
        ocr_count = 0

        # First pass: determine body font size (most common size across all pages)
        body_size = self._detect_body_font_size(pdf)
        pdf.close()

        # Process in chunks of 10 pages to bound memory usage.
        # pdfplumber holds per-page layout data in memory; for large PDFs
        # (100+ pages) this can OOM. Re-opening per chunk keeps it bounded.
        _CHUNK_SIZE = 10
        for chunk_start in range(0, page_count, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, page_count)
            chunk_pdf = pdfplumber.open(file.local_path)

            for page_idx in range(chunk_start, chunk_end):
                page = chunk_pdf.pages[page_idx]
                page_num = page_idx + 1
                chars = page.chars

                if len(chars) < _MIN_TEXT_CHARS:
                    try:
                        page_elements = self._ocr_page_to_elements(file, page_idx)
                        ocr_count += 1
                    except Exception as e:
                        logger.warning(f"GLM-OCR failed for page {page_num}: {e}")
                        page_elements = []
                else:
                    page_elements = self._extract_page_elements(page, page_num, body_size)

                elements.extend(page_elements)
                page.flush_cache()

            chunk_pdf.close()

        if ocr_count > 0:
            logger.info(
                f"Parsed {page_count} pages ({ocr_count} via GLM-OCR, "
                f"{page_count - ocr_count} via pdfplumber)"
            )

        # Post-processing: fix no-space titles, filter noise, deduplicate
        elements = self._post_process_elements(elements)

        return ParsedDocument(
            source=file.uri,
            elements=elements,
            metadata=self._build_metadata(
                file, page_count=page_count, parser="glm_ocr", ocr_pages=ocr_count
            ),
        )

    def _detect_body_font_size(self, pdf: Any) -> float:
        """Find the most common font size across the document (= body text)."""
        size_counts: Counter = Counter()
        for page in pdf.pages[:10]:  # Sample first 10 pages
            for char in page.chars:
                size_counts[round(char["size"], 1)] += 1
        if not size_counts:
            return 10.0
        return float(size_counts.most_common(1)[0][0])

    def _post_process_elements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """Post-process: fix no-space titles, filter noise headings, deduplicate."""
        result = []
        seen_headings: set[str] = set()

        for el in elements:
            if el.type == ElementType.HEADING:
                # 1. Fix no-space camelCase titles: "RiskMeasurement" → "Risk Measurement"
                title = _fix_nospace_title(el.content)

                # 2. Filter noise headings
                if _is_noise_heading(title):
                    # Demote to TEXT instead of dropping (content may be useful)
                    result.append(
                        DocumentElement(
                            type=ElementType.TEXT,
                            content=title,
                            page=el.page,
                            metadata=el.metadata,
                        )
                    )
                    continue

                # 3. Deduplicate headings (same title = skip)
                dedup_key = title.lower().strip()
                if dedup_key in seen_headings:
                    continue
                seen_headings.add(dedup_key)

                # Replace with cleaned title
                result.append(
                    DocumentElement(
                        type=ElementType.HEADING,
                        content=title,
                        level=el.level,
                        page=el.page,
                        metadata=el.metadata,
                    )
                )
            else:
                result.append(el)

        return result

    def _extract_page_elements(
        self, page: Any, page_num: int, body_size: float
    ) -> List[DocumentElement]:
        """Extract structured elements using pdfplumber word-level font info.

        Heading detection uses three signals (any one is sufficient):
        1. Font size larger than body text
        2. Bold font (fontname contains 'Bold' or 'Medi')
        3. Section heading pattern (Roman numerals, numbered sections)
        """
        words = page.extract_words(extra_attrs=["size", "fontname"])
        if not words:
            return []

        # Group words into lines by Y-position (same top = same line)
        lines = self._group_words_into_lines(words)

        # Merge consecutive lines of the same type into paragraphs
        elements: List[DocumentElement] = []
        current_text = ""
        current_size = 0.0
        current_is_heading = False

        for line_text, line_size, line_bold in lines:
            # A line is a heading if any of: larger font, bold, or matches section pattern
            is_heading = (
                line_size > body_size + 0.5
                or (line_bold and len(line_text) < 120)
                or bool(_SECTION_HEADING_RE.match(line_text))
            )

            if current_text and is_heading != current_is_heading:
                elements.append(
                    self._make_element(
                        current_text.strip(), current_size, body_size, page_num, current_is_heading
                    )
                )
                current_text = line_text
                current_size = line_size
                current_is_heading = is_heading
            else:
                sep = " " if current_text and not current_text.endswith("\n") else ""
                current_text += sep + line_text
                current_size = max(current_size, line_size)
                if is_heading:
                    current_is_heading = True

        if current_text.strip():
            elements.append(
                self._make_element(
                    current_text.strip(), current_size, body_size, page_num, current_is_heading
                )
            )

        return elements

    def _group_words_into_lines(self, words: list[dict]) -> list[tuple[str, float, bool]]:
        """Group words into lines by Y-position.

        Returns list of (line_text, max_font_size, is_bold).
        """
        if not words:
            return []

        def _is_bold(fontname: str) -> bool:
            fn = fontname.lower()
            return "bold" in fn or "medi" in fn

        lines: list[tuple[str, float, bool]] = []
        current_words: list[str] = [words[0]["text"]]
        current_size = words[0]["size"]
        current_top = words[0]["top"]
        current_bold = _is_bold(words[0].get("fontname", ""))

        for word in words[1:]:
            if abs(word["top"] - current_top) < current_size * 0.5:
                current_words.append(word["text"])
                current_size = max(current_size, word["size"])
                if _is_bold(word.get("fontname", "")):
                    current_bold = True
            else:
                lines.append((" ".join(current_words), round(current_size, 1), current_bold))
                current_words = [word["text"]]
                current_size = word["size"]
                current_top = word["top"]
                current_bold = _is_bold(word.get("fontname", ""))

        lines.append((" ".join(current_words), round(current_size, 1), current_bold))
        return lines

    def _make_element(
        self,
        text: str,
        font_size: float,
        body_size: float,
        page_num: int,
        force_heading: bool = False,
    ) -> DocumentElement:
        """Create a DocumentElement, classifying as HEADING or TEXT.

        Uses font size, bold detection, and section patterns.
        """
        is_heading_by_size = font_size > body_size + 0.5
        is_heading = (force_heading or is_heading_by_size) and (
            len(text) < 200
            and len(text) > 3
            and not text.startswith("Page ")
            and not text.strip().replace(".", "").replace(",", "").isdigit()
        )
        if is_heading:
            # Heading level from font size (if available) or default L2 for bold/pattern
            size_diff = font_size - body_size
            if size_diff > 8:
                level = 1
            elif size_diff > 4:
                level = 2
            else:
                level = 3
            return DocumentElement(
                type=ElementType.HEADING,
                content=text,
                level=level,
                page=page_num,
                metadata={"page": page_num, "font_size": font_size},
            )
        return DocumentElement(
            type=ElementType.TEXT,
            content=text,
            page=page_num,
            metadata={"page": page_num},
        )

    def _parse_image(self, file: SourceFile) -> ParsedDocument:
        """Parse a single image file via GLM-OCR."""
        image_bytes = self._read_file_bytes(file)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        markdown = self._ocr_image(image_b64)
        elements = self._markdown_to_elements(markdown, page=1)

        return ParsedDocument(
            source=file.uri,
            elements=elements,
            metadata=self._build_metadata(file, parser="glm_ocr"),
        )

    def _ocr_page_to_elements(self, file: SourceFile, page_idx: int) -> List[DocumentElement]:
        """Render a specific PDF page to image and OCR it with GLM-OCR."""
        import pypdfium2 as pdfium

        file_bytes = self._read_file_bytes(file)
        pdf = pdfium.PdfDocument(file_bytes)
        page = pdf[page_idx]

        bitmap = page.render(scale=self.dpi / 72)
        pil_image = bitmap.to_pil()
        pdf.close()

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        markdown = self._ocr_image(image_b64)
        return self._markdown_to_elements(markdown, page=page_idx + 1)

    def _ocr_image(self, image_base64: str) -> str:
        """Send image to GLM-OCR via ollama and return text."""
        import httpx

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": "Text Recognition:",
                "images": [image_base64],
                "stream": False,
                "options": {"num_ctx": 16384},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return str(response.json().get("response", ""))

    # --- GLM-OCR markdown output parsing (for OCR path) ---

    def _markdown_to_elements(self, markdown: str, page: int) -> List[DocumentElement]:
        """Parse GLM-OCR markdown output into DocumentElements."""
        elements = []
        lines = markdown.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                content = heading_match.group(2).strip()
                elements.append(
                    DocumentElement(
                        type=ElementType.HEADING,
                        content=content,
                        level=level,
                        page=page,
                        metadata={"page": page},
                    )
                )
                i += 1
                continue

            if _CODE_FENCE_RE.match(line):
                lang = line.strip("`").strip() or None
                code_lines = []
                i += 1
                while i < len(lines) and not _CODE_FENCE_RE.match(lines[i]):
                    code_lines.append(lines[i])
                    i += 1
                i += 1
                elements.append(
                    DocumentElement(
                        type=ElementType.CODE_BLOCK,
                        content="\n".join(code_lines),
                        language=lang,
                        page=page,
                        metadata={"page": page},
                    )
                )
                continue

            if _TABLE_ROW_RE.match(line.strip()):
                table_lines = []
                while i < len(lines) and _TABLE_ROW_RE.match(lines[i].strip()):
                    if not re.match(r"^\|[\s\-:]+\|$", lines[i].strip()):
                        table_lines.append(lines[i])
                    i += 1
                elements.append(
                    DocumentElement(
                        type=ElementType.TABLE,
                        content="\n".join(table_lines),
                        page=page,
                        metadata={"page": page},
                    )
                )
                continue

            if line.strip():
                para_lines = [line]
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not _HEADING_RE.match(lines[i])
                    and not _CODE_FENCE_RE.match(lines[i])
                    and not _TABLE_ROW_RE.match(lines[i].strip())
                ):
                    para_lines.append(lines[i])
                    i += 1
                elements.append(
                    DocumentElement(
                        type=ElementType.TEXT,
                        content="\n".join(para_lines).strip(),
                        page=page,
                        metadata={"page": page},
                    )
                )
                continue

            i += 1

        return elements
