# tests/test_ingest_hashing.py
"""
Tests for fitz_ai.ingest.hashing module.
"""

from pathlib import Path

import pytest

from fitz_ai.ingest.hashing import (
    compute_bytes_hash,
    compute_chunk_id,
    compute_content_hash,
)


class TestComputeContentHash:
    """Tests for compute_content_hash."""

    def test_computes_hash_for_file(self, tmp_path: Path):
        """Test that hash is computed correctly for a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")

        result = compute_content_hash(test_file)

        assert result.startswith("sha256:")
        assert len(result) == 71  # "sha256:" + 64 hex chars

    def test_same_content_same_hash(self, tmp_path: Path):
        """Test that identical content produces identical hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        content = "identical content"
        file1.write_text(content, encoding="utf-8")
        file2.write_text(content, encoding="utf-8")

        hash1 = compute_content_hash(file1)
        hash2 = compute_content_hash(file2)

        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path: Path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content A", encoding="utf-8")
        file2.write_text("content B", encoding="utf-8")

        hash1 = compute_content_hash(file1)
        hash2 = compute_content_hash(file2)

        assert hash1 != hash2

    def test_raises_for_nonexistent_file(self, tmp_path: Path):
        """Test that FileNotFoundError is raised for missing file."""
        missing = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            compute_content_hash(missing)

    def test_raises_for_directory(self, tmp_path: Path):
        """Test that IsADirectoryError is raised for directory."""
        with pytest.raises(IsADirectoryError):
            compute_content_hash(tmp_path)

    def test_handles_binary_file(self, tmp_path: Path):
        """Test that binary files are handled correctly."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")

        result = compute_content_hash(binary_file)

        assert result.startswith("sha256:")

    def test_handles_large_file(self, tmp_path: Path):
        """Test that large files are handled correctly."""
        large_file = tmp_path / "large.txt"
        # Write 1MB of data
        large_file.write_bytes(b"x" * (1024 * 1024))

        result = compute_content_hash(large_file)

        assert result.startswith("sha256:")


class TestComputeBytesHash:
    """Tests for compute_bytes_hash."""

    def test_computes_hash_for_bytes(self):
        """Test that hash is computed correctly for bytes."""
        result = compute_bytes_hash(b"hello world")

        assert result.startswith("sha256:")
        assert len(result) == 71

    def test_same_bytes_same_hash(self):
        """Test that identical bytes produce identical hash."""
        data = b"identical"
        hash1 = compute_bytes_hash(data)
        hash2 = compute_bytes_hash(data)

        assert hash1 == hash2

    def test_empty_bytes(self):
        """Test that empty bytes produce valid hash."""
        result = compute_bytes_hash(b"")

        assert result.startswith("sha256:")


class TestComputeChunkId:
    """Tests for compute_chunk_id."""

    def test_deterministic(self):
        """Test that same inputs produce same ID."""
        id1 = compute_chunk_id(
            content_hash="sha256:abc",
            chunk_index=0,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )
        id2 = compute_chunk_id(
            content_hash="sha256:abc",
            chunk_index=0,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )

        assert id1 == id2

    def test_different_content_hash_different_id(self):
        """Test that different content hash produces different ID."""
        id1 = compute_chunk_id(
            content_hash="sha256:abc",
            chunk_index=0,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )
        id2 = compute_chunk_id(
            content_hash="sha256:def",
            chunk_index=0,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )

        assert id1 != id2

    def test_different_chunk_index_different_id(self):
        """Test that different chunk index produces different ID."""
        id1 = compute_chunk_id(
            content_hash="sha256:abc",
            chunk_index=0,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )
        id2 = compute_chunk_id(
            content_hash="sha256:abc",
            chunk_index=1,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )

        assert id1 != id2

    def test_different_config_different_id(self):
        """Test that different config produces different ID."""
        base = {
            "content_hash": "sha256:abc",
            "chunk_index": 0,
            "parser_id": "md.v1",
            "chunker_id": "tokens_800_120",
            "embedding_id": "openai:text-embedding-3-small",
        }

        id1 = compute_chunk_id(**base)

        # Different parser
        id2 = compute_chunk_id(**{**base, "parser_id": "md.v2"})
        assert id1 != id2

        # Different chunker
        id3 = compute_chunk_id(**{**base, "chunker_id": "tokens_1000_150"})
        assert id1 != id3

        # Different embedding
        id4 = compute_chunk_id(**{**base, "embedding_id": "cohere:embed-english-v3.0"})
        assert id1 != id4

    def test_returns_hex_string(self):
        """Test that result is a valid hex string."""
        result = compute_chunk_id(
            content_hash="sha256:abc",
            chunk_index=0,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )

        # Should be 64 hex chars (no prefix)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
