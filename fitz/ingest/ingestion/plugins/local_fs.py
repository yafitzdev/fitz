# fitz/ingest/ingestion/plugins/local_fs.py
"""
Local filesystem ingestion plugin.

Reads files from local filesystem with automatic encoding detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

from fitz.ingest.ingestion.base import RawDocument

logger = logging.getLogger(__name__)


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
    if raw_bytes.startswith(b'\xff\xfe'):
        # UTF-16 LE BOM
        return raw_bytes.decode('utf-16-le')
    elif raw_bytes.startswith(b'\xfe\xff'):
        # UTF-16 BE BOM
        return raw_bytes.decode('utf-16-be')
    elif raw_bytes.startswith(b'\xef\xbb\xbf'):
        # UTF-8 BOM
        return raw_bytes[3:].decode('utf-8', errors='ignore')

    # Check for null bytes (sign of UTF-16 without BOM)
    if b'\x00' in raw_bytes[:100]:
        # Likely UTF-16
        try:
            # Try UTF-16 LE first (more common on Windows)
            return raw_bytes.decode('utf-16-le')
        except UnicodeDecodeError:
            try:
                return raw_bytes.decode('utf-16-be')
            except UnicodeDecodeError:
                pass

    # Try UTF-8 first (most common)
    try:
        return raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        pass

    # Fallback to latin-1 (never fails)
    return raw_bytes.decode('latin-1')


class LocalFSIngestPlugin:
    """
    Local filesystem ingestion plugin.

    Reads text files from a directory or single file path.
    Automatically detects file encoding (UTF-8, UTF-16, etc.).
    """

    plugin_name = "local"

    def __init__(self, **_: Any) -> None:
        pass

    def ingest(self, source: str, kwargs: Dict[str, Any]) -> Iterable[RawDocument]:
        base = Path(source)

        paths = [base] if base.is_file() else list(base.glob("**/*"))

        for path in paths:
            if not path.is_file():
                continue

            try:
                content = _read_text_with_encoding_detection(path)
            except Exception as e:
                logger.warning(f"Skipping {path}: {type(e).__name__}: {e}")
                continue

            yield RawDocument(
                path=str(path),
                content=content,
                metadata={
                    "source": "local_fs",
                    **(kwargs.get("metadata") or {}),
                },
            )