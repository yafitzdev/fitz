# tests/unit/test_content_reader.py
"""Tests for ContentReader: line range extraction, symbol reading."""

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
from fitz_sage.engines.fitz_krag.types import Address, AddressKind


@pytest.fixture
def mock_raw_store():
    store = MagicMock()
    store.get.return_value = {
        "id": "f1",
        "path": "src/module.py",
        "content": "import os\n\ndef hello():\n    return 'world'\n\ndef goodbye():\n    return 'bye'\n",
        "content_hash": "abc",
        "file_type": ".py",
        "size_bytes": 100,
        "metadata": {},
    }
    return store


class TestContentReader:
    def test_read_symbol(self, mock_raw_store):
        reader = ContentReader(mock_raw_store)
        addr = Address(
            kind=AddressKind.SYMBOL,
            source_id="f1",
            location="module.hello",
            summary="Say hello",
            metadata={"start_line": 3, "end_line": 4},
        )
        results = reader.read([addr], limit=5)
        assert len(results) == 1
        assert "def hello():" in results[0].content
        assert results[0].line_range == (3, 4)
        assert results[0].file_path == "src/module.py"

    def test_read_file(self, mock_raw_store):
        reader = ContentReader(mock_raw_store)
        addr = Address(
            kind=AddressKind.FILE,
            source_id="f1",
            location="src/module.py",
            summary="Module file",
        )
        results = reader.read([addr], limit=5)
        assert len(results) == 1
        assert "import os" in results[0].content
        assert results[0].line_range is None

    def test_read_missing_file(self):
        store = MagicMock()
        store.get.return_value = None
        reader = ContentReader(store)
        addr = Address(
            kind=AddressKind.SYMBOL,
            source_id="missing",
            location="x.y",
            summary="test",
            metadata={"start_line": 1, "end_line": 2},
        )
        results = reader.read([addr], limit=5)
        assert len(results) == 0

    def test_respects_limit(self, mock_raw_store):
        reader = ContentReader(mock_raw_store)
        addrs = [
            Address(
                kind=AddressKind.SYMBOL,
                source_id="f1",
                location=f"mod.func{i}",
                summary=f"func {i}",
                metadata={"start_line": 1, "end_line": 2},
            )
            for i in range(10)
        ]
        results = reader.read(addrs, limit=3)
        assert len(results) == 3

    def test_read_section_returns_none_without_store(self, mock_raw_store):
        reader = ContentReader(mock_raw_store)
        addr = Address(
            kind=AddressKind.SECTION,
            source_id="f1",
            location="Section 1",
            summary="A section",
            metadata={"section_id": "sec1"},
        )
        results = reader.read([addr], limit=5)
        assert len(results) == 0

    def test_read_section_with_store(self, mock_raw_store):
        mock_section_store = MagicMock()
        mock_section_store.get.return_value = {
            "id": "sec1",
            "raw_file_id": "f1",
            "title": "Introduction",
            "level": 1,
            "page_start": 3,
            "page_end": 5,
            "content": "This is the introduction section.",
            "summary": "Intro summary.",
            "parent_section_id": None,
            "position": 0,
            "metadata": {},
        }
        reader = ContentReader(mock_raw_store, section_store=mock_section_store)
        addr = Address(
            kind=AddressKind.SECTION,
            source_id="f1",
            location="Introduction",
            summary="Intro summary.",
            metadata={"section_id": "sec1"},
        )
        results = reader.read([addr], limit=5)
        assert len(results) == 1
        assert results[0].content == "This is the introduction section."
        assert results[0].file_path == "src/module.py"
        assert results[0].metadata["page_start"] == 3
        assert results[0].metadata["page_end"] == 5
        assert results[0].metadata["section_title"] == "Introduction"
