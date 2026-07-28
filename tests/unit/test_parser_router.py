"""Tests for deterministic document-parser routing."""

from __future__ import annotations

import pytest

from fitz_sage.ingestion.parser.plugins.cpu_pdf import CpuPdfParser
from fitz_sage.ingestion.parser.plugins.lightweight import (
    LightweightDOCXParser,
    LightweightPPTXParser,
)
from fitz_sage.ingestion.parser.router import ParserRouter


def test_cpu_mode_uses_lightweight_parsers_even_when_docling_is_installed() -> None:
    router = ParserRouter(parser="cpu")

    assert isinstance(router.get_parser(".pdf"), CpuPdfParser)
    assert isinstance(router.get_parser(".docx"), LightweightDOCXParser)
    assert isinstance(router.get_parser(".pptx"), LightweightPPTXParser)


def test_unknown_parser_mode_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown parser mode"):
        ParserRouter(parser="automatic")
