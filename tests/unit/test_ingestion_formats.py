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
        parser="cpu",
    )

    assert DOCUMENT_EXTENSIONS - {".xlsx"} <= extensions
    assert ".xlsx" not in extensions
    assert {".py", ".ts", ".tsx", ".js", ".jsx", ".csv", ".tsv"} <= extensions
    assert ".java" not in extensions
    assert ".cs" not in extensions
    assert ".env" not in extensions


def test_binary_formats_follow_the_selected_parser_contract() -> None:
    docling = enabled_extensions(
        code_languages=set(),
        table_extensions=set(),
        parser="docling",
    )
    glm_ocr = enabled_extensions(
        code_languages=set(),
        table_extensions=set(),
        parser="glm_ocr",
    )

    assert {".pdf", ".docx", ".pptx", ".xlsx"} <= docling
    assert ".pdf" in glm_ocr
    assert not ({".docx", ".pptx", ".xlsx"} & glm_ocr)
