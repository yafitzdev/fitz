# tests/test_chunker_id.py
"""
Tests for chunker_id behavior.

Verifies:
1. Each chunker generates a deterministic chunker_id
2. chunker_id changes when parameters change
3. chunker_id format follows the convention: "{plugin_name}:{param1}:{param2}:..."
"""

import pytest

from fitz_ai.ingest.chunking.plugins.simple import SimpleChunker


class TestSimpleChunkerID:
    """Tests for SimpleChunker.chunker_id"""

    def test_default_chunker_id(self):
        """Default parameters produce expected ID."""
        chunker = SimpleChunker()
        assert chunker.chunker_id == "simple:1000:0"

    def test_custom_chunk_size(self):
        """Custom chunk_size is reflected in ID."""
        chunker = SimpleChunker(chunk_size=500)
        assert chunker.chunker_id == "simple:500:0"

    def test_custom_overlap(self):
        """Custom chunk_overlap is reflected in ID."""
        chunker = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunker_id == "simple:1000:100"

    def test_full_custom_id(self):
        """Both parameters reflected in ID."""
        chunker = SimpleChunker(chunk_size=800, chunk_overlap=120)
        assert chunker.chunker_id == "simple:800:120"

    def test_id_is_deterministic(self):
        """Same parameters always produce same ID."""
        chunker1 = SimpleChunker(chunk_size=500, chunk_overlap=50)
        chunker2 = SimpleChunker(chunk_size=500, chunk_overlap=50)
        assert chunker1.chunker_id == chunker2.chunker_id

    def test_different_params_different_id(self):
        """Different parameters produce different IDs."""
        chunker1 = SimpleChunker(chunk_size=500)
        chunker2 = SimpleChunker(chunk_size=600)
        assert chunker1.chunker_id != chunker2.chunker_id

    def test_plugin_name_in_id(self):
        """Plugin name is the first component of the ID."""
        chunker = SimpleChunker()
        assert chunker.chunker_id.startswith("simple:")

    def test_id_format_is_colon_separated(self):
        """ID uses colon as separator."""
        chunker = SimpleChunker(chunk_size=800, chunk_overlap=100)
        parts = chunker.chunker_id.split(":")
        assert len(parts) == 3
        assert parts[0] == "simple"
        assert parts[1] == "800"
        assert parts[2] == "100"


class TestChunkerValidation:
    """Tests for parameter validation."""

    def test_invalid_chunk_size_zero(self):
        """chunk_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            SimpleChunker(chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        """Negative chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            SimpleChunker(chunk_size=-100)

    def test_invalid_overlap_negative(self):
        """Negative chunk_overlap raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            SimpleChunker(chunk_overlap=-1)

    def test_overlap_exceeds_chunk_size(self):
        """chunk_overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be < chunk_size"):
            SimpleChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap .* must be < chunk_size"):
            SimpleChunker(chunk_size=100, chunk_overlap=150)


class TestChunkerIDForRechunking:
    """Tests verifying chunker_id changes trigger re-chunking."""

    def test_size_change_triggers_rechunk(self):
        """Changing chunk_size produces different ID."""
        original = SimpleChunker(chunk_size=1000, chunk_overlap=0)
        modified = SimpleChunker(chunk_size=800, chunk_overlap=0)
        assert original.chunker_id != modified.chunker_id

    def test_overlap_change_triggers_rechunk(self):
        """Changing chunk_overlap produces different ID."""
        original = SimpleChunker(chunk_size=1000, chunk_overlap=0)
        modified = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        assert original.chunker_id != modified.chunker_id

    def test_same_config_no_rechunk(self):
        """Same configuration produces same ID."""
        original = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        same = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        assert original.chunker_id == same.chunker_id


class TestChunkingBehavior:
    """Tests for actual chunking output."""

    def test_simple_chunking_basic(self):
        """Basic chunking produces expected chunks."""
        chunker = SimpleChunker(chunk_size=10, chunk_overlap=0)
        text = "0123456789ABCDEFGHIJ"

        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) == 2
        assert chunks[0].content == "0123456789"
        assert chunks[1].content == "ABCDEFGHIJ"

    def test_overlap_chunking(self):
        """Overlap chunking includes overlapping content."""
        chunker = SimpleChunker(chunk_size=10, chunk_overlap=3)
        text = "0123456789ABCDEFGHIJ"

        chunks = chunker.chunk_text(text, {"doc_id": "test"})

        assert len(chunks) == 3
        assert chunks[0].content == "0123456789"
        assert chunks[1].content == "789ABCDEFG"
        assert chunks[2].content == "EFGHIJ"

    def test_empty_text_returns_empty_list(self):
        """Empty text produces no chunks."""
        chunker = SimpleChunker()
        assert chunker.chunk_text("", {"doc_id": "test"}) == []
        assert chunker.chunk_text("   ", {"doc_id": "test"}) == []

    def test_chunk_metadata(self):
        """Chunks include base metadata."""
        chunker = SimpleChunker(chunk_size=100)
        meta = {"doc_id": "mydoc", "source_file": "/path/to/file.txt"}

        chunks = chunker.chunk_text("Hello world", meta)

        assert len(chunks) == 1
        assert chunks[0].doc_id == "mydoc"
        assert chunks[0].metadata["source_file"] == "/path/to/file.txt"

    def test_chunk_ids_are_sequential(self):
        """Chunk IDs are sequential within a document."""
        chunker = SimpleChunker(chunk_size=5, chunk_overlap=0)
        text = "AAAAABBBBBCCCCC"

        chunks = chunker.chunk_text(text, {"doc_id": "doc1"})

        assert chunks[0].id == "doc1:0"
        assert chunks[1].id == "doc1:1"
        assert chunks[2].id == "doc1:2"
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1
        assert chunks[2].chunk_index == 2
