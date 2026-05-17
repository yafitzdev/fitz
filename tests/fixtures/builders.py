# tests/fixtures/builders.py
"""
Test fixture builders using the Builder pattern.

Instead of using MagicMock everywhere, these builders create realistic
test objects with sensible defaults that can be customized as needed.

This reduces mock references and makes tests more reliable by testing
against real implementations rather than mock behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.ingestion.source.base import SourceFile

# =============================================================================
# Common Test Data Builders
# =============================================================================


class SourceFileBuilder:
    """Builder for SourceFile test fixtures."""

    def __init__(self):
        self._uri = "file:///test/file.txt"
        self._local_path = Path("/test/file.txt")
        self._metadata = {}

    def with_uri(self, uri: str) -> SourceFileBuilder:
        self._uri = uri
        return self

    def with_path(self, path: str | Path) -> SourceFileBuilder:
        self._local_path = Path(path)
        return self

    def with_extension(self, ext: str) -> SourceFileBuilder:
        """Set file extension (updates path)."""
        if not ext.startswith("."):
            ext = f".{ext}"
        self._local_path = self._local_path.with_suffix(ext)
        return self

    def with_metadata(self, **kwargs) -> SourceFileBuilder:
        self._metadata.update(kwargs)
        return self

    def build(self) -> SourceFile:
        return SourceFile(
            uri=self._uri,
            local_path=self._local_path,
            metadata=self._metadata,
        )


class DocumentElementBuilder:
    """Builder for DocumentElement test fixtures."""

    def __init__(self):
        self._type = ElementType.TEXT
        self._content = "Test content"
        self._metadata = {}
        self._level = None
        self._language = None

    def as_text(self, content: str) -> DocumentElementBuilder:
        self._type = ElementType.TEXT
        self._content = content
        return self

    def as_heading(self, content: str, level: int = 1) -> DocumentElementBuilder:
        self._type = ElementType.HEADING
        self._content = content
        self._level = level
        return self

    def as_code(self, content: str, language: str = "python") -> DocumentElementBuilder:
        self._type = ElementType.CODE_BLOCK
        self._content = content
        self._language = language
        return self

    def as_table(self, content: str) -> DocumentElementBuilder:
        self._type = ElementType.TABLE
        self._content = content
        return self

    def with_metadata(self, **kwargs) -> DocumentElementBuilder:
        self._metadata.update(kwargs)
        return self

    def build(self) -> DocumentElement:
        return DocumentElement(
            type=self._type,
            content=self._content,
            metadata=self._metadata,
            level=self._level,
            language=self._language,
        )


class ParsedDocumentBuilder:
    """Builder for ParsedDocument test fixtures."""

    def __init__(self):
        self._source = "file:///test/doc.txt"
        self._elements = []
        self._metadata = {}
        self._tables = []

    def with_source(self, source: str) -> ParsedDocumentBuilder:
        self._source = source
        return self

    def with_element(self, element: DocumentElement) -> ParsedDocumentBuilder:
        self._elements.append(element)
        return self

    def with_text(self, content: str) -> ParsedDocumentBuilder:
        """Shortcut to add a text element."""
        element = DocumentElementBuilder().as_text(content).build()
        self._elements.append(element)
        return self

    def with_heading(self, content: str, level: int = 1) -> ParsedDocumentBuilder:
        """Shortcut to add a heading element."""
        element = DocumentElementBuilder().as_heading(content, level).build()
        self._elements.append(element)
        return self

    def with_code(self, content: str, language: str = "python") -> ParsedDocumentBuilder:
        """Shortcut to add a code element."""
        element = DocumentElementBuilder().as_code(content, language).build()
        self._elements.append(element)
        return self

    def with_metadata(self, **kwargs) -> ParsedDocumentBuilder:
        self._metadata.update(kwargs)
        return self

    def build(self) -> ParsedDocument:
        # Add default element if none provided
        if not self._elements:
            self._elements = [DocumentElementBuilder().build()]

        return ParsedDocument(
            source=self._source,
            elements=self._elements,
            metadata=self._metadata,
            tables=self._tables,
        )


class AddressBuilder:
    """Builder for Address test fixtures."""

    def __init__(self):
        self._kind = AddressKind.CHUNK
        self._source_id = "test_source"
        self._location = "chunk_0"
        self._summary = "Test summary"
        self._score = 0.0
        self._metadata = {}

    def as_chunk(self, chunk_id: str = "chunk_0") -> AddressBuilder:
        self._kind = AddressKind.CHUNK
        self._location = chunk_id
        return self

    def as_symbol(self, qualified_name: str) -> AddressBuilder:
        self._kind = AddressKind.SYMBOL
        self._location = qualified_name
        self._metadata["qualified_name"] = qualified_name
        return self

    def as_raw(self, file_path: str, line_start: int, line_end: int) -> AddressBuilder:
        self._kind = AddressKind.RAW
        self._location = f"{file_path}:{line_start}-{line_end}"
        self._metadata["file_path"] = file_path
        self._metadata["line_start"] = line_start
        self._metadata["line_end"] = line_end
        return self

    def with_summary(self, summary: str) -> AddressBuilder:
        self._summary = summary
        return self

    def with_score(self, score: float) -> AddressBuilder:
        self._score = score
        return self

    def with_metadata(self, **kwargs) -> AddressBuilder:
        self._metadata.update(kwargs)
        return self

    def build(self) -> Address:
        return Address(
            kind=self._kind,
            source_id=self._source_id,
            location=self._location,
            summary=self._summary,
            score=self._score,
            metadata=self._metadata,
        )


class ReadResultBuilder:
    """Builder for ReadResult test fixtures."""

    def __init__(self):
        self._address = AddressBuilder().build()
        self._content = "Test content"
        self._file_path = "test.py"
        self._line_range = (1, 10)

    def with_address(self, address: Address) -> ReadResultBuilder:
        self._address = address
        return self

    def with_content(self, content: str) -> ReadResultBuilder:
        self._content = content
        return self

    def with_file(self, path: str, start_line: int = 1, end_line: int = 10) -> ReadResultBuilder:
        self._file_path = path
        self._line_range = (start_line, end_line)
        return self

    def with_score(self, score: float) -> ReadResultBuilder:
        self._address.score = score
        return self

    def for_symbol(self, name: str, content: str = None) -> ReadResultBuilder:
        """Shortcut for creating a symbol result."""
        self._address = AddressBuilder().as_symbol(f"module.{name}").build()
        self._content = content or f"def {name}():\n    pass"
        return self

    def for_class_method(self, class_name: str, method_name: str) -> ReadResultBuilder:
        """Shortcut for creating a class method result."""
        qualified = f"module.{class_name}.{method_name}"
        self._address = (
            AddressBuilder()
            .as_symbol(qualified)
            .with_metadata(kind="method", class_name=class_name)
            .build()
        )
        self._content = f"def {method_name}(self):\n    pass"
        return self

    def build(self) -> ReadResult:
        return ReadResult(
            address=self._address,
            content=self._content,
            file_path=self._file_path,
            line_range=self._line_range,
        )


# =============================================================================
# Store Implementations for Testing
# =============================================================================


@dataclass
class InMemoryRawStore:
    """Simple in-memory implementation of RawStore for testing."""

    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_file(self, file_id: str, path: str, content: str) -> None:
        """Add a file to the store."""
        self.files[file_id] = {
            "id": file_id,
            "path": path,
            "content": content,
        }

    def get(self, file_id: str) -> Dict[str, Any] | None:
        """Get a file by ID."""
        return self.files.get(file_id)


@dataclass
class InMemorySymbolStore:
    """Simple in-memory implementation of SymbolStore for testing."""

    symbols: List[Dict[str, Any]] = field(default_factory=list)

    def add_symbol(
        self,
        qualified_name: str,
        kind: str = "function",
        file_id: str = "f1",
        start_line: int = 1,
        end_line: int = 10,
    ) -> None:
        """Add a symbol to the store."""
        self.symbols.append(
            {
                "qualified_name": qualified_name,
                "kind": kind,
                "file_id": file_id,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    def get_class_symbols(self, file_id: str, class_name: str) -> List[Dict[str, Any]]:
        """Get symbols for a class."""
        return [
            s
            for s in self.symbols
            if s["file_id"] == file_id and s["qualified_name"].startswith(f"module.{class_name}.")
        ]


@dataclass
class InMemoryImportStore:
    """Simple in-memory implementation of ImportStore for testing."""

    imports: Dict[str, List[str]] = field(default_factory=dict)

    def add_imports(self, file_id: str, imported_files: List[str]) -> None:
        """Add import relationships."""
        self.imports[file_id] = imported_files

    def get_imported_files(self, file_id: str) -> List[str]:
        """Get files imported by the given file."""
        return self.imports.get(file_id, [])


# =============================================================================
# Collection/Config Builders
# =============================================================================


class CollectionInfoBuilder:
    """Builder for CollectionInfo test fixtures."""

    def __init__(self):
        self._name = "test_collection"
        self._chunk_count = 100
        self._metadata = {}

    def with_name(self, name: str) -> CollectionInfoBuilder:
        self._name = name
        return self

    def with_chunks(self, count: int) -> CollectionInfoBuilder:
        self._chunk_count = count
        return self

    def empty(self) -> CollectionInfoBuilder:
        """Create an empty collection."""
        self._chunk_count = 0
        return self

    def build(self):
        from fitz_sage.services.fitz_service import CollectionInfo

        return CollectionInfo(
            name=self._name,
            chunk_count=self._chunk_count,
            metadata=self._metadata,
        )


# =============================================================================
# Factory Functions (shortcuts for common patterns)
# =============================================================================


def make_text_file(extension: str = ".txt") -> SourceFile:
    """Create a simple text file fixture."""
    return SourceFileBuilder().with_extension(extension).build()


def make_parsed_doc(content: str = "Test document") -> ParsedDocument:
    """Create a simple parsed document fixture."""
    return ParsedDocumentBuilder().with_text(content).build()


def make_symbol_result(name: str = "test_func") -> ReadResult:
    """Create a symbol read result fixture."""
    return ReadResultBuilder().for_symbol(name).build()


def make_test_stores() -> tuple[InMemoryRawStore, InMemorySymbolStore, InMemoryImportStore]:
    """Create a set of in-memory stores for testing."""
    return InMemoryRawStore(), InMemorySymbolStore(), InMemoryImportStore()
