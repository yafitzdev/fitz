# tests/test_sdk_fitz.py
"""
Tests for the Fitz SDK.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestFitzInit:
    """Tests for fitz initialization."""

    def test_default_collection(self):
        """Test default collection name."""
        from fitz_ai.sdk import fitz

        f = fitz()

        assert f.collection == "default"

    def test_custom_collection(self):
        """Test custom collection name."""
        from fitz_ai.sdk import fitz

        f = fitz(collection="my_collection")

        assert f.collection == "my_collection"

    def test_custom_config_path(self, tmp_path):
        """Test custom config path."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "custom.yaml"
        f = fitz(config_path=config_path)

        assert f.config_path == config_path

    def test_auto_init_default_true(self):
        """Test auto_init defaults to True."""
        from fitz_ai.sdk import fitz

        f = fitz()
        # auto_init is an internal attribute
        assert f._auto_init is True


class TestFitzConfigCreation:
    """Tests for config file creation."""

    def test_ensure_config_creates_file(self, tmp_path):
        """Test that _ensure_config creates config when auto_init=True."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / ".fitz" / "config.yaml"
        f = fitz(config_path=config_path, auto_init=True)

        # Call ensure config
        f._ensure_config()

        # Config should have been created
        assert config_path.exists()

    def test_config_has_required_sections(self, tmp_path):
        """Test that created config has all required sections."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path, collection="test")

        # Manually call the config creation
        f._create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        assert "chat" in config
        assert "embedding" in config
        assert "vector_db" in config
        assert "retrieval" in config
        assert "rerank" in config
        assert "rgs" in config

    def test_config_uses_collection_name(self, tmp_path):
        """Test that config uses the fitz instance's collection name."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path, collection="my_collection")

        f._create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        assert config["retrieval"]["collection"] == "my_collection"

    def test_raises_without_auto_init(self, tmp_path):
        """Test that ConfigurationError is raised when auto_init=False and no config."""
        from fitz_ai.core import ConfigurationError
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "nonexistent.yaml"
        f = fitz(config_path=config_path, auto_init=False)

        with pytest.raises(ConfigurationError):
            f._ensure_config()


class TestFitzIngest:
    """Tests for fitz.ingest() method."""

    def test_raises_on_nonexistent_source(self, tmp_path):
        """Test that ValueError is raised for nonexistent source."""
        from fitz_ai.sdk import fitz

        f = fitz(config_path=tmp_path / "config.yaml")

        with pytest.raises(ValueError, match="does not exist"):
            f.ingest("/nonexistent/path")

    def test_raises_on_empty_source(self, tmp_path):
        """Test that ValueError is raised when no documents found."""
        from fitz_ai.sdk import fitz

        config_path = tmp_path / "config.yaml"
        f = fitz(config_path=config_path)

        # Create empty docs directory
        docs_path = tmp_path / "empty_docs"
        docs_path.mkdir()

        # Create config first
        f._create_default_config(config_path)

        with patch("fitz_ai.ingest.ingestion.registry.get_ingest_plugin") as mock_ingest:
            mock_plugin_cls = MagicMock()
            mock_plugin_instance = MagicMock()
            mock_plugin_cls.return_value = mock_plugin_instance
            mock_ingest.return_value = mock_plugin_cls

            with patch("fitz_ai.ingest.ingestion.engine.IngestionEngine") as mock_engine:
                mock_engine_instance = MagicMock()
                mock_engine_instance.run.return_value = []  # No documents
                mock_engine.return_value = mock_engine_instance

                with pytest.raises(ValueError, match="No documents found"):
                    f.ingest(docs_path)


class TestFitzAsk:
    """Tests for fitz.ask() method."""

    def test_raises_on_empty_question(self, tmp_path):
        """Test that ValueError is raised for empty question."""
        from fitz_ai.sdk import fitz

        f = fitz(config_path=tmp_path / "config.yaml")

        with pytest.raises(ValueError, match="cannot be empty"):
            f.ask("")

    def test_raises_on_whitespace_question(self, tmp_path):
        """Test that ValueError is raised for whitespace-only question."""
        from fitz_ai.sdk import fitz

        f = fitz(config_path=tmp_path / "config.yaml")

        with pytest.raises(ValueError, match="cannot be empty"):
            f.ask("   ")


class TestFitzQuery:
    """Tests for fitz.query() alias."""

    def test_query_is_alias_for_ask(self, tmp_path):
        """Test that query() is an alias for ask()."""
        from fitz_ai.sdk import fitz

        f = fitz(config_path=tmp_path / "config.yaml")

        # Both should raise the same error
        with pytest.raises(ValueError, match="cannot be empty"):
            f.query("")


class TestIngestStats:
    """Tests for IngestStats dataclass."""

    def test_ingest_stats_fields(self):
        """Test IngestStats has expected fields."""
        from fitz_ai.sdk import IngestStats

        stats = IngestStats(documents=10, chunks=50, collection="test")

        assert stats.documents == 10
        assert stats.chunks == 50
        assert stats.collection == "test"


class TestFitzExports:
    """Tests for SDK exports."""

    def test_fitz_exported_from_sdk(self):
        """Test fitz is exported from fitz_ai.sdk."""
        from fitz_ai.sdk import fitz

        assert fitz is not None

    def test_fitz_exported_from_top_level(self):
        """Test fitz is exported from fitz_ai."""
        from fitz_ai import fitz

        assert fitz is not None

    def test_ingest_stats_exported_from_top_level(self):
        """Test IngestStats is exported from fitz_ai."""
        from fitz_ai import IngestStats

        assert IngestStats is not None
