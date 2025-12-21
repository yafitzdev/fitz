# fitz_ai/ingest/ingestion/plugins/local_fs.py
"""
Local filesystem ingestion plugin.

Reads files from local filesystem with automatic encoding detection.
Supports PDF text extraction via pdfplumber (preferred) or pypdf (fallback).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from fitz_ai.ingest.ingestion.base import RawDocument

logger = logging.getLogger(__name__)


# =============================================================================
# PDF Text Extraction
# =============================================================================


def _extract_pdf_text_pdfplumber(path: Path) -> Optional[str]:
    """
    Extract text from PDF using pdfplumber (best quality).

    Returns None if pdfplumber is not installed.
    """
    try:
        import pdfplumber
    except ImportError:
        return None

    try:
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        return "\n\n".join(text_parts) if text_parts else ""
    except Exception as e:
        logger.warning(f"pdfplumber failed on {path}: {e}")
        return None


def _extract_pdf_text_pypdf(path: Path) -> Optional[str]:
    """
    Extract text from PDF using pypdf (fallback).

    Returns None if pypdf is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n\n".join(text_parts) if text_parts else ""
    except Exception as e:
        logger.warning(f"pypdf failed on {path}: {e}")
        return None


def _extract_pdf_text(path: Path) -> str:
    """
    Extract text from a PDF file.

    Tries pdfplumber first (better quality), falls back to pypdf.
    Raises RuntimeError if no PDF library is available.
    """
    # Try pdfplumber first (better text extraction)
    text = _extract_pdf_text_pdfplumber(path)
    if text is not None:
        return text

    # Fall back to pypdf
    text = _extract_pdf_text_pypdf(path)
    if text is not None:
        return text

    # No PDF library available
    raise RuntimeError(
        f"Cannot extract text from PDF: {path}. "
        "Install pdfplumber (recommended) or pypdf: "
        "pip install pdfplumber  OR  pip install pypdf"
    )


# =============================================================================
# Text File Reading
# =============================================================================


def _read_text_with_encoding_detection(path: Path) -> str:
    """
    Read text file with automatic encoding detection.

    Handles:
    - UTF-8 (default)
    - UTF-8 with BOM
    - UTF-16 LE (Windows PowerShell default)
    - UTF-16 BE
    - Latin-1 fallback
    """
    # Read raw bytes first
    raw_bytes = path.read_bytes()

    # Check for BOM and detect encoding
    if raw_bytes.startswith(b"\xff\xfe"):
        # UTF-16 LE BOM
        return raw_bytes.decode("utf-16-le")
    elif raw_bytes.startswith(b"\xfe\xff"):
        # UTF-16 BE BOM
        return raw_bytes.decode("utf-16-be")
    elif raw_bytes.startswith(b"\xef\xbb\xbf"):
        # UTF-8 BOM
        return raw_bytes[3:].decode("utf-8", errors="ignore")

    # Check for null bytes (sign of UTF-16 without BOM)
    if b"\x00" in raw_bytes[:100]:
        # Likely UTF-16
        try:
            # Try UTF-16 LE first (more common on Windows)
            return raw_bytes.decode("utf-16-le")
        except UnicodeDecodeError:
            try:
                return raw_bytes.decode("utf-16-be")
            except UnicodeDecodeError:
                pass

    # Try UTF-8 first (most common)
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Fallback to latin-1 (never fails)
    return raw_bytes.decode("latin-1")


# =============================================================================
# Supported File Types
# =============================================================================


# File extensions that are treated as text files
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
    ".cmd",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".log",
    ".ini",
    ".cfg",
    ".conf",
    ".toml",
    ".tex",
    ".bib",
}

# File extensions that require special handling
PDF_EXTENSIONS = {".pdf"}

# File extensions to skip (binary files, images, etc.)
SKIP_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wav",
    ".flac",
    ".doc",
    ".xls",
    ".ppt",  # Old Office formats (need specialized handling)
    ".docx",
    ".xlsx",
    ".pptx",  # New Office formats (need specialized handling)
    ".pyc",
    ".pyo",
    ".class",
    ".db",
    ".sqlite",
    ".sqlite3",
}


def _get_file_type(path: Path) -> str:
    """
    Determine file type based on extension.

    Returns: "text", "pdf", "skip", or "unknown"
    """
    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return "text"
    elif ext in PDF_EXTENSIONS:
        return "pdf"
    elif ext in SKIP_EXTENSIONS:
        return "skip"
    else:
        return "unknown"


# =============================================================================
# Main Plugin
# =============================================================================


class LocalFSIngestPlugin:
    """
    Local filesystem ingestion plugin.

    Reads files from a directory or single file path.

    Supported file types:
    - Text files (.txt, .md, .py, etc.) - with automatic encoding detection
    - PDF files (.pdf) - text extraction via pdfplumber or pypdf

    Files with unknown extensions are attempted as text files.
    Binary files and media files are skipped.
    """

    plugin_name = "local"

    def __init__(self, **_: Any) -> None:
        pass

    def ingest(self, source: str, kwargs: Dict[str, Any]) -> Iterable[RawDocument]:
        base = Path(source)

        if base.is_file():
            paths = [base]
        else:
            paths = list(base.glob("**/*"))

        for path in paths:
            if not path.is_file():
                continue

            file_type = _get_file_type(path)

            # Skip binary/media files
            if file_type == "skip":
                logger.debug(f"Skipping binary/media file: {path}")
                continue

            try:
                if file_type == "pdf":
                    # Extract text from PDF
                    content = _extract_pdf_text(path)
                    logger.info(f"Extracted {len(content)} chars from PDF: {path.name}")
                else:
                    # Read as text file (text or unknown)
                    content = _read_text_with_encoding_detection(path)

                # Skip empty files
                if not content or not content.strip():
                    logger.debug(f"Skipping empty file: {path}")
                    continue

            except Exception as e:
                logger.warning(f"Skipping {path}: {type(e).__name__}: {e}")
                continue

            yield RawDocument(
                path=str(path),
                content=content,
                metadata={
                    "source": "local_fs",
                    "file_type": file_type,
                    "file_extension": path.suffix.lower(),
                    **(kwargs.get("metadata") or {}),
                },
            )
