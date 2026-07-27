# fitz_sage/engines/fitz_krag/progressive/parsed_cache.py
"""
Parsed text cache for rich documents (PDF, DOCX, PPTX, HTML).

Caches parsed text by content hash so documents are only parsed once.
Cache lives at .fitz/collections/{col}/parsed/{content_hash}.txt.

Also caches heading structure from Docling parses:
    .fitz/collections/{col}/parsed/{content_hash}.headings.json

PDF strategy:
  1. Try fast native text extraction via pypdfium2 (<1s for 100 pages)
  2. If PDF has embedded text (born-digital), use it directly
  3. If scanned/image PDF (no extractable text), fall back to Docling+OCR
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum average chars per page to consider a PDF as having extractable text
_MIN_CHARS_PER_PAGE = 50


def get_parsed_text(
    path: Path,
    content_hash: str,
    cache_dir: Path,
) -> str | None:
    """Get parsed text for a rich document, using cache when available.

    Args:
        path: Absolute path to the file on disk.
        content_hash: SHA-256 hash of the file's raw bytes.
        cache_dir: Directory to store/read cached parsed text.

    Returns:
        Parsed text, or None if parsing fails.
    """
    # Check cache first
    cache_path = cache_dir / f"{content_hash}.txt"
    if cache_path.exists():
        try:
            cached_text = cache_path.read_text(encoding="utf-8")
            if cached_text.strip():
                return cached_text
        except Exception:
            pass  # Cache corrupt, re-parse

    # Parse — try fast extraction for PDFs, Docling for everything else
    if path.suffix.lower() == ".pdf":
        parsed_text, headings = _parse_pdf_fast(path)
    else:
        parsed_text, headings = _parse_docling(path)

    if not parsed_text:
        return None

    # Save to cache
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(parsed_text, encoding="utf-8")
    except Exception as e:
        logger.debug(f"Failed to cache parsed text for {path.name}: {e}")

    # Save headings to cache (if Docling produced them)
    if headings:
        try:
            headings_path = cache_dir / f"{content_hash}.headings.json"
            headings_path.write_text(json.dumps(headings), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to cache headings for {path.name}: {e}")

    return parsed_text


def parse_document_text(path: Path) -> str | None:
    """Parse a rich document without persisting a cache entry."""
    if path.suffix.lower() == ".pdf":
        text, _ = _parse_pdf_fast(path)
    else:
        text, _ = _parse_docling(path)
    return text


def get_parsed_headings(content_hash: str, cache_dir: Path) -> list[dict[str, object]]:
    """Read cached headings for a rich document.

    Returns:
        List of {"title": str, "level": int} dicts, or empty list.
    """
    headings_path = cache_dir / f"{content_hash}.headings.json"
    if not headings_path.exists():
        return []
    try:
        data = json.loads(headings_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _parse_pdf_fast(path: Path) -> tuple[str | None, list[dict] | None]:
    """Extract text from PDF using fast native extraction, falling back to Docling.

    Born-digital PDFs (LaTeX, Word-exported, etc.) have embedded text that
    can be extracted in <1s. Only scanned/image PDFs need OCR via Docling.

    Returns:
        (text, headings) — headings is None for pypdfium2 fast path.
    """
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        n_pages = len(pdf)

        pages: list[str] = []
        for i in range(n_pages):
            page = pdf[i]
            text = page.get_textpage().get_text_bounded()
            if text and text.strip():
                pages.append(text.strip())

        pdf.close()

        total_chars = sum(len(p) for p in pages)
        avg_chars = total_chars / max(n_pages, 1)

        if avg_chars >= _MIN_CHARS_PER_PAGE:
            logger.debug(
                f"Fast PDF extract: {n_pages} pages, {total_chars} chars, "
                f"{avg_chars:.0f} avg chars/page"
            )
            return "\n\n".join(pages), None

        # Too little text — likely scanned/image PDF, fall back to Docling
        logger.debug(
            f"PDF has low text density ({avg_chars:.0f} chars/page), "
            f"falling back to Docling+OCR"
        )
    except Exception as e:
        logger.debug(f"Fast PDF extraction failed, falling back to Docling: {e}")

    return _parse_docling(path)


_parser_logs_suppressed = False


def _parse_docling(path: Path) -> tuple[str | None, list[dict] | None]:
    """Parse a rich document using Docling (with OCR for scanned PDFs).

    Returns:
        (text, headings) — headings extracted from Docling's element structure.
    """
    global _parser_logs_suppressed
    try:
        # Globally disable INFO logs during parsing — RapidOCR overrides
        # its logger level during init, so per-logger suppression doesn't work.
        import logging as _logging

        if not _parser_logs_suppressed:
            _logging.disable(_logging.INFO)

        from fitz_sage.core.document import ElementType
        from fitz_sage.ingestion.parser import ParserRouter
        from fitz_sage.ingestion.source.base import SourceFile

        source_file = SourceFile(uri=path.as_uri(), local_path=path)
        router = ParserRouter()
        parsed = router.parse(source_file)

        if not _parser_logs_suppressed:
            _logging.disable(_logging.NOTSET)
            _suppress_parser_logs()
            _parser_logs_suppressed = True

        parsed_text: str | None = parsed.full_text
        parsed_text = parsed_text if parsed_text and parsed_text.strip() else None

        # Extract headings from structured elements
        headings: list[dict[str, object]] | None = None
        if parsed.elements:
            heading_list = [
                {"title": el.content.strip(), "level": el.level or 1}
                for el in parsed.elements
                if el.type == ElementType.HEADING and el.content and el.content.strip()
            ]
            if heading_list:
                headings = heading_list

        return parsed_text, headings
    except Exception as e:
        if not _parser_logs_suppressed:
            import logging as _logging

            _logging.disable(_logging.NOTSET)
        logger.warning(f"Parser failed for {path.name}: {e}")
        return None, None


def _suppress_parser_logs() -> None:
    """Suppress noisy third-party logs from PDF/doc parsers permanently."""
    import logging as _logging

    for name in ("rapidocr", "docling", "RapidOCR", "fitz_sage.ingestion.parser"):
        _logger = _logging.getLogger(name)
        _logger.setLevel(_logging.WARNING)
        _logger.handlers.clear()
