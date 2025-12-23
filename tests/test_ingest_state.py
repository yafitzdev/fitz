# tests/test_ingest_state.py
"""
Tests for fitz_ai.ingest.state module.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path

from fitz_ai.ingest.state import (
    IngestStateManager,
    IngestState,
    FileEntry,
    FileStatus,
    EmbeddingConfig,
    ChunkingConfigEntry,
)


class TestIngestState:
    """Tests for IngestState model."""

    def test_create_default_state(self):
        """Test creating a default state."""
        state = IngestState(project_id="test-123")

        assert state.schema_version == 1
        assert state.project_id == "test-123"
        assert state.roots == {}
        assert ".md" in state.parsing_by_ext
        assert ".md" in state.chunking_by_ext

    def test_get_parser_id(self):
        """Test getting parser ID for known and unknown extensions."""
        state = IngestState(project_id="test")

        assert state.get_parser_id(".md") == "md.v1"
        assert state.get_parser_id(".txt") == "txt.v1"
        assert state.get_parser_id(".unknown") == "unknown.v1"

    def test_get_chunker_id(self):
        """Test getting chunker ID for known and unknown extensions."""
        state = IngestState(project_id="test")

        assert state.get_chunker_id(".md") == "tokens_800_120"
        assert state.get_chunker_id(".unknown") == "tokens_800_120"

    def test_get_embedding_id(self):
        """Test getting embedding ID."""
        state = IngestState(project_id="test")
        assert state.get_embedding_id() == "unknown:unknown"

        state.embedding = EmbeddingConfig.create("openai", "text-embedding-3-small")
        assert state.get_embedding_id() == "openai:text-embedding-3-small"

    def test_get_active_paths(self):
        """Test getting active paths from state."""
        state = IngestState(project_id="test")

        # No paths initially
        assert state.get_active_paths("/root") == set()

    def test_get_file_entry(self):
        """Test getting file entry."""
        state = IngestState(project_id="test")

        # No entry initially
        assert state.get_file_entry("/root", "/root/file.md") is None


class TestFileEntry:
    """Tests for FileEntry model."""

    def test_is_active(self):
        """Test is_active method."""
        active = FileEntry(
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
            status=FileStatus.ACTIVE,
        )
        assert active.is_active() is True

        deleted = FileEntry(
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
            status=FileStatus.DELETED,
        )
        assert deleted.is_active() is False


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_create(self):
        """Test creating embedding config."""
        config = EmbeddingConfig.create("openai", "text-embedding-3-small")

        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.id == "openai:text-embedding-3-small"

    def test_create_with_extras(self):
        """Test creating embedding config with dimension."""
        config = EmbeddingConfig.create(
            "openai",
            "text-embedding-3-small",
            dimension=1536,
            normalize=True,
        )

        assert config.dimension == 1536
        assert config.normalize is True


class TestChunkingConfigEntry:
    """Tests for ChunkingConfigEntry model."""

    def test_create(self):
        """Test creating chunking config."""
        config = ChunkingConfigEntry.create("tokens", 800, 120)

        assert config.strategy == "tokens"
        assert config.chunk_size == 800
        assert config.overlap == 120
        assert config.id == "tokens_800_120"


class TestIngestStateManager:
    """Tests for IngestStateManager."""

    def test_creates_new_state_if_missing(self, tmp_path: Path):
        """Test that new state is created if file doesn't exist."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)

        state = manager.load()

        assert state.schema_version == 1
        assert len(state.project_id) > 0

    def test_loads_existing_state(self, tmp_path: Path):
        """Test loading an existing state file."""
        state_path = tmp_path / "ingest.json"

        # Create a state file
        state_data = {
            "schema_version": 1,
            "project_id": "existing-project",
            "updated_at": "2024-01-01T00:00:00",
            "roots": {},
            "chunking_by_ext": {},
            "parsing_by_ext": {},
        }
        state_path.write_text(json.dumps(state_data))

        manager = IngestStateManager(state_path)
        state = manager.load()

        assert state.project_id == "existing-project"

    def test_mark_active(self, tmp_path: Path):
        """Test marking a file as active."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        manager.mark_active(
            file_path="/root/test.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
        )

        entry = manager.get_file_entry("/root", "/root/test.md")
        assert entry is not None
        assert entry.content_hash == "sha256:abc123"
        assert entry.status == FileStatus.ACTIVE

    def test_mark_deleted(self, tmp_path: Path):
        """Test marking a file as deleted."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        # First mark as active
        manager.mark_active(
            file_path="/root/test.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
        )

        # Then mark as deleted
        manager.mark_deleted("/root/test.md", "/root")

        entry = manager.get_file_entry("/root", "/root/test.md")
        assert entry is not None
        assert entry.status == FileStatus.DELETED

    def test_save_and_reload(self, tmp_path: Path):
        """Test saving and reloading state."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        manager.mark_active(
            file_path="/root/test.md",
            root="/root",
            content_hash="sha256:abc123",
            ext=".md",
            size_bytes=1234,
            mtime_epoch=1234567890.0,
        )
        manager.save()

        # Create new manager and load
        manager2 = IngestStateManager(state_path)
        manager2.load()

        entry = manager2.get_file_entry("/root", "/root/test.md")
        assert entry is not None
        assert entry.content_hash == "sha256:abc123"

    def test_get_active_paths(self, tmp_path: Path):
        """Test getting active paths."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        # Add some files
        manager.mark_active("/root/a.md", "/root", "sha256:a", ".md", 100, 1234567890.0)
        manager.mark_active("/root/b.md", "/root", "sha256:b", ".md", 100, 1234567890.0)
        manager.mark_active("/root/c.md", "/root", "sha256:c", ".md", 100, 1234567890.0)

        # Delete one
        manager.mark_deleted("/root/b.md", "/root")

        active = manager.get_active_paths("/root")
        assert active == {"/root/a.md", "/root/c.md"}

    def test_set_embedding_config(self, tmp_path: Path):
        """Test setting embedding config."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        manager.set_embedding_config("openai", "text-embedding-3-small")

        assert manager.get_embedding_id() == "openai:text-embedding-3-small"