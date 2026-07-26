"""Tests for the authoritative KRAG file-format contract."""

from fitz_sage.engines.fitz_krag.ingestion.formats import (
    BINARY_DOCUMENT_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    RICH_DOCUMENT_EXTENSIONS,
    enabled_extensions,
)


def test_rich_document_contract_matches_parser_backed_formats() -> None:
    assert {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"} <= (RICH_DOCUMENT_EXTENSIONS)
    assert ".html" not in BINARY_DOCUMENT_EXTENSIONS
    assert {".pdf", ".docx", ".pptx", ".xlsx"} == BINARY_DOCUMENT_EXTENSIONS


def test_enabled_extensions_are_explicit_and_configuration_aware() -> None:
    extensions = enabled_extensions(
        code_languages={"python", "typescript"},
        table_extensions={"csv", ".tsv"},
    )

    assert DOCUMENT_EXTENSIONS <= extensions
    assert {".py", ".ts", ".tsx", ".js", ".jsx", ".csv", ".tsv"} <= extensions
    assert ".java" not in extensions
    assert ".cs" not in extensions
    assert ".env" not in extensions
