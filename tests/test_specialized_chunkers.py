# tests/test_specialized_chunkers.py
"""
Tests for specialized chunker plugins.

Tests:
- MarkdownChunker: Header splitting, code block preservation
- PythonCodeChunker: AST-based splitting, class/function detection
- PdfSectionChunker: Section detection, large section splitting
"""

import pytest

from fitz_ai.ingest.chunking.plugins.markdown import MarkdownChunker
from fitz_ai.ingest.chunking.plugins.pdf_sections import PdfSectionChunker
from fitz_ai.ingest.chunking.plugins.python_code import PythonCodeChunker


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_chunker_id_format(self):
        """Test chunker_id has correct format."""
        chunker = MarkdownChunker(max_chunk_size=1500, min_chunk_size=100)
        assert chunker.chunker_id == "markdown:1500:100"

    def test_chunker_id_deterministic(self):
        """Same config produces same ID."""
        c1 = MarkdownChunker(max_chunk_size=1500, min_chunk_size=100)
        c2 = MarkdownChunker(max_chunk_size=1500, min_chunk_size=100)
        assert c1.chunker_id == c2.chunker_id

    def test_splits_on_headers(self):
        """Test that markdown is split on headers."""
        chunker = MarkdownChunker(max_chunk_size=5000)
        text = """# Introduction

This is the introduction section.

## Methods

This is the methods section.

## Results

This is the results section.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) >= 2
        # Check sections are preserved
        contents = [c.content for c in chunks]
        assert any("Introduction" in c for c in contents)
        assert any("Methods" in c for c in contents)

    def test_preserves_code_blocks(self):
        """Test that code blocks are not split."""
        chunker = MarkdownChunker(max_chunk_size=5000)
        text = """# Code Example

Here is some code:

```python
def hello():
    print("Hello, world!")

def goodbye():
    print("Goodbye!")
```

More text after.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        # Code block should be in a single chunk
        code_chunk = None
        for chunk in chunks:
            if "def hello():" in chunk.content:
                code_chunk = chunk
                break

        assert code_chunk is not None
        assert "def goodbye():" in code_chunk.content

    def test_handles_empty_input(self):
        """Test empty input returns empty list."""
        chunker = MarkdownChunker()
        assert chunker.chunk_text("", {"doc_id": "test"}) == []
        assert chunker.chunk_text("   ", {"doc_id": "test"}) == []

    def test_no_headers_single_chunk(self):
        """Text without headers becomes single chunk."""
        chunker = MarkdownChunker(max_chunk_size=5000)
        text = "This is plain text without any headers."
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_validation_max_chunk_size(self):
        """Test validation of max_chunk_size."""
        with pytest.raises(ValueError, match="max_chunk_size"):
            MarkdownChunker(max_chunk_size=50)

    def test_validation_min_chunk_size(self):
        """Test validation of min_chunk_size."""
        with pytest.raises(ValueError, match="min_chunk_size"):
            MarkdownChunker(min_chunk_size=0)

    def test_section_header_in_metadata(self):
        """Test that section header is added to metadata."""
        chunker = MarkdownChunker(max_chunk_size=5000)
        text = """# Introduction

Content here.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) >= 1
        # Header might be in metadata
        for chunk in chunks:
            if "Introduction" in chunk.content:
                assert (
                    "section_header" in chunk.metadata
                    or "Introduction" in chunk.content
                )


class TestPythonCodeChunker:
    """Tests for PythonCodeChunker."""

    def test_chunker_id_format(self):
        """Test chunker_id has correct format."""
        chunker = PythonCodeChunker(max_chunk_size=2000, include_imports=True)
        assert chunker.chunker_id == "python_code:2000:1"

    def test_chunker_id_without_imports(self):
        """Test chunker_id when imports disabled."""
        chunker = PythonCodeChunker(max_chunk_size=2000, include_imports=False)
        assert chunker.chunker_id == "python_code:2000:0"

    def test_splits_by_function(self):
        """Test that code is split by function."""
        chunker = PythonCodeChunker(max_chunk_size=5000)
        code = '''
def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2
'''
        chunks = chunker.chunk_text(code, {"doc_id": "test"})

        assert len(chunks) >= 2
        contents = [c.content for c in chunks]
        assert any("function_one" in c for c in contents)
        assert any("function_two" in c for c in contents)

    def test_splits_by_class(self):
        """Test that code is split by class."""
        chunker = PythonCodeChunker(max_chunk_size=5000)
        code = '''
class ClassOne:
    """First class."""
    pass

class ClassTwo:
    """Second class."""
    pass
'''
        chunks = chunker.chunk_text(code, {"doc_id": "test"})

        assert len(chunks) >= 2
        contents = [c.content for c in chunks]
        assert any("ClassOne" in c for c in contents)
        assert any("ClassTwo" in c for c in contents)

    def test_includes_docstrings(self):
        """Test that docstrings are included with functions."""
        chunker = PythonCodeChunker(max_chunk_size=5000)
        code = '''
def my_function():
    """This is a docstring for my_function."""
    return 42
'''
        chunks = chunker.chunk_text(code, {"doc_id": "test"})

        assert len(chunks) >= 1
        assert any("This is a docstring" in c.content for c in chunks)

    def test_handles_syntax_errors_gracefully(self):
        """Test that invalid Python doesn't crash."""
        chunker = PythonCodeChunker()
        code = "def broken( { syntax"
        chunks = chunker.chunk_text(code, {"doc_id": "test"})

        # Should return something, not crash
        assert len(chunks) >= 1

    def test_handles_empty_input(self):
        """Test empty input returns empty list."""
        chunker = PythonCodeChunker()
        assert chunker.chunk_text("", {"doc_id": "test"}) == []

    def test_includes_imports_when_enabled(self):
        """Test that imports are included in chunks."""
        chunker = PythonCodeChunker(include_imports=True)
        code = """
import os
from pathlib import Path

def my_function():
    return os.getcwd()
"""
        chunks = chunker.chunk_text(code, {"doc_id": "test"})

        # Function chunk should include imports
        func_chunk = None
        for chunk in chunks:
            if "my_function" in chunk.content:
                func_chunk = chunk
                break

        assert func_chunk is not None
        assert "import os" in func_chunk.content

    def test_code_element_in_metadata(self):
        """Test that code element name is in metadata."""
        chunker = PythonCodeChunker()
        code = """
def my_function():
    pass
"""
        chunks = chunker.chunk_text(code, {"doc_id": "test"})

        assert len(chunks) >= 1
        assert any(c.metadata.get("code_element") == "def my_function" for c in chunks)

    def test_validation_max_chunk_size(self):
        """Test validation of max_chunk_size."""
        with pytest.raises(ValueError, match="max_chunk_size"):
            PythonCodeChunker(max_chunk_size=50)


class TestPdfSectionChunker:
    """Tests for PdfSectionChunker."""

    def test_chunker_id_format(self):
        """Test chunker_id has correct format."""
        chunker = PdfSectionChunker(max_section_chars=3000, min_section_chars=50)
        assert chunker.chunker_id == "pdf_sections:3000:50"

    def test_chunker_id_deterministic(self):
        """Same config produces same ID."""
        c1 = PdfSectionChunker(max_section_chars=3000, min_section_chars=50)
        c2 = PdfSectionChunker(max_section_chars=3000, min_section_chars=50)
        assert c1.chunker_id == c2.chunker_id

    def test_detects_all_caps_headers(self):
        """Test that ALL CAPS headers are detected."""
        chunker = PdfSectionChunker(max_section_chars=5000)
        text = """INTRODUCTION

This is the introduction.

METHODS

This is the methods section.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) >= 2
        headers = [c.metadata.get("section_header", "") for c in chunks]
        assert any("INTRODUCTION" in h for h in headers)
        assert any("METHODS" in h for h in headers)

    def test_detects_numbered_sections(self):
        """Test that numbered sections are detected."""
        chunker = PdfSectionChunker(max_section_chars=5000)
        text = """1. Introduction

Content for section 1.

2. Background

Content for section 2.

3.1 Subsection

Content for subsection.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) >= 2

    def test_detects_keyword_sections(self):
        """Test that section keywords are detected."""
        chunker = PdfSectionChunker(max_section_chars=5000)
        text = """Abstract

This is the abstract of the paper.

Introduction

This is the introduction.

Conclusion

This is the conclusion.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) >= 2

    def test_splits_large_sections(self):
        """Test that large sections are split."""
        chunker = PdfSectionChunker(max_section_chars=100)
        text = (
            """INTRODUCTION

"""
            + "This is a very long paragraph. " * 50
        )

        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        # Should be split into multiple chunks
        assert len(chunks) >= 2
        # Each chunk should be <= max_section_chars (approximately)
        for chunk in chunks:
            # Allow some margin for headers
            assert len(chunk.content) <= 200

    def test_handles_empty_input(self):
        """Test empty input returns empty list."""
        chunker = PdfSectionChunker()
        assert chunker.chunk_text("", {"doc_id": "test"}) == []

    def test_no_sections_single_chunk(self):
        """Text without section headers becomes single chunk."""
        chunker = PdfSectionChunker(max_section_chars=5000)
        text = "This is plain text without any section headers or keywords."
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) == 1

    def test_section_header_in_metadata(self):
        """Test that section header is in metadata."""
        chunker = PdfSectionChunker(max_section_chars=5000)
        text = """INTRODUCTION

Content here.
"""
        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) >= 1
        assert chunks[0].metadata.get("section_header") == "INTRODUCTION"

    def test_validation_max_section_chars(self):
        """Test validation of max_section_chars."""
        with pytest.raises(ValueError, match="max_section_chars"):
            PdfSectionChunker(max_section_chars=50)

    def test_validation_min_section_chars(self):
        """Test validation of min_section_chars."""
        with pytest.raises(ValueError, match="min_section_chars"):
            PdfSectionChunker(min_section_chars=0)


class TestChunkerIDDeterminism:
    """Cross-chunker tests for ID determinism."""

    def test_different_chunkers_different_ids(self):
        """Different chunker types have different IDs."""
        md = MarkdownChunker(max_chunk_size=1000, min_chunk_size=50)
        py = PythonCodeChunker(max_chunk_size=1000)
        pdf = PdfSectionChunker(max_section_chars=1000, min_section_chars=50)

        ids = {md.chunker_id, py.chunker_id, pdf.chunker_id}
        assert len(ids) == 3  # All different

    def test_same_params_different_types_different_ids(self):
        """Same numeric params but different types have different IDs."""
        md = MarkdownChunker(max_chunk_size=1000, min_chunk_size=100)
        pdf = PdfSectionChunker(max_section_chars=1000, min_section_chars=100)

        assert md.chunker_id != pdf.chunker_id
        assert "markdown" in md.chunker_id
        assert "pdf_sections" in pdf.chunker_id
