"""Authoritative file-format contract for KRAG ingestion."""

from __future__ import annotations

from collections.abc import Iterable

CODE_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".java": "java",
    ".go": "go",
}

# Fitz parses these as documents. It does not normalize identifiers or infer
# domain meaning from their contents.
PLAIN_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".cfg",
        ".conf",
        ".gql",
        ".graphql",
        ".ini",
        ".json",
        ".jsonl",
        ".md",
        ".rst",
        ".sql",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
RICH_DOCUMENT_EXTENSIONS = frozenset({".docx", ".htm", ".html", ".pdf", ".pptx", ".xlsx"})
BINARY_DOCUMENT_EXTENSIONS = frozenset({".docx", ".pdf", ".pptx", ".xlsx"})
DOCUMENT_EXTENSIONS = PLAIN_DOCUMENT_EXTENSIONS | RICH_DOCUMENT_EXTENSIONS


def enabled_extensions(
    *,
    code_languages: Iterable[str],
    table_extensions: Iterable[str],
    parser: str = "cpu",
) -> frozenset[str]:
    """Return normalized extensions enabled by one KRAG configuration."""
    languages = {str(language).strip().lower() for language in code_languages}
    code = {
        extension for extension, language in CODE_EXTENSION_MAP.items() if language in languages
    }
    tables = {_normalize_extension(extension) for extension in table_extensions}
    documents = set(DOCUMENT_EXTENSIONS)
    parser_mode = str(parser).strip().lower()
    if parser_mode == "cpu":
        documents.discard(".xlsx")
    elif parser_mode == "glm_ocr":
        documents -= {".docx", ".pptx", ".xlsx"}
    return frozenset(code | documents | tables)


def _normalize_extension(extension: str) -> str:
    normalized = str(extension).strip().lower()
    return normalized if normalized.startswith(".") else f".{normalized}"


__all__ = [
    "CODE_EXTENSION_MAP",
    "BINARY_DOCUMENT_EXTENSIONS",
    "DOCUMENT_EXTENSIONS",
    "PLAIN_DOCUMENT_EXTENSIONS",
    "RICH_DOCUMENT_EXTENSIONS",
    "enabled_extensions",
]
