# tests/test_ingest_state.py
"""
Tests for fitz_ai.ingest.state module.
"""

import json
from pathlib import Path


from fitz_ai.ingest.state import (
    EmbeddingConfig,
    FileEntry,
    FileStatus,
    IngestState,
    IngestStateManager,
)


class TestIngestState:
    """Tests for IngestState model."""

    def test_create_default_state(self):
        """Test creating a default state."""
        state = IngestState(project_id="test-123")

        assert state.schema_version == 1
        assert state.project_id == "test-123"
        assert state.roots == {}

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
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )
        assert active.is_active() is True

        deleted = FileEntry(
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
            status=FileStatus.DELETED,
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
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
        )

        assert config.dimension == 1536


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
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        entry = manager.get_file_entry("/root", "/root/test.md")
        assert entry is not None
        assert entry.content_hash == "sha256:abc123"
        assert entry.status == FileStatus.ACTIVE
        assert entry.chunker_id == "simple:1000:0"
        assert entry.parser_id == "md.v1"
        assert entry.embedding_id == "cohere:embed-english-v3.0"

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
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        # Then mark as deleted
        manager.mark_deleted("/root", "/root/test.md")

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
            chunker_id="simple:1000:0",
            parser_id="md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )
        manager.save()

        # Create new manager and load
        manager2 = IngestStateManager(state_path)
        manager2.load()

        entry = manager2.get_file_entry("/root", "/root/test.md")
        assert entry is not None
        assert entry.content_hash == "sha256:abc123"
        assert entry.chunker_id == "simple:1000:0"

    def test_get_active_paths(self, tmp_path: Path):
        """Test getting active paths."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        # Add some files
        manager.mark_active(
            "/root/a.md",
            "/root",
            "sha256:a",
            ".md",
            100,
            1234567890.0,
            "simple:1000:0",
            "md.v1",
            "cohere:embed-english-v3.0",
        )
        manager.mark_active(
            "/root/b.md",
            "/root",
            "sha256:b",
            ".md",
            100,
            1234567890.0,
            "simple:1000:0",
            "md.v1",
            "cohere:embed-english-v3.0",
        )
        manager.mark_active(
            "/root/c.md",
            "/root",
            "sha256:c",
            ".md",
            100,
            1234567890.0,
            "simple:1000:0",
            "md.v1",
            "cohere:embed-english-v3.0",
        )

        # Delete one
        manager.mark_deleted("/root", "/root/b.md")

        active = manager.get_active_paths("/root")
        assert active == {"/root/a.md", "/root/c.md"}

    def test_set_embedding_config(self, tmp_path: Path):
        """Test setting embedding config."""
        state_path = tmp_path / "ingest.json"
        manager = IngestStateManager(state_path)
        manager.load()

        manager.set_embedding_config("openai", "text-embedding-3-small")

        assert manager.state.embedding is not None
        assert manager.state.embedding.id == "openai:text-embedding-3-small"

    def test_config_ids_persisted(self, tmp_path: Path):
        """Test that config IDs are persisted and loaded correctly."""
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
            chunker_id="markdown:800:100",
            parser_id="md.v2",
            embedding_id="openai:text-embedding-3-large",
        )
        manager.save()

        # Reload and verify
        manager2 = IngestStateManager(state_path)
        manager2.load()

        entry = manager2.get_file_entry("/root", "/root/test.md")
        assert entry.chunker_id == "markdown:800:100"
        assert entry.parser_id == "md.v2"
        assert entry.embedding_id == "openai:text-embedding-3-large"
