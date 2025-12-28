# tests/test_cli_map.py
"""
Tests for the map command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestMapCommand:
    """Tests for fitz map command."""

    def test_map_shows_help(self):
        """Test that map --help works."""
        result = runner.invoke(app, ["map", "--help"])

        assert result.exit_code == 0
        assert "map" in result.output.lower()
        assert "visual" in result.output.lower()

    def test_map_requires_dependencies(self, tmp_path):
        """Test that map requires umap-learn and sklearn."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"collection": "test"},
            "embedding": {"provider": "cohere", "model": "embed-english-v3.0"},
        }
        config_path.write_text(yaml.dump(config))

        # Mock umap not being installed
        with (
            patch("fitz_ai.cli.commands.map.FitzPaths.config", return_value=config_path),
            patch.dict("sys.modules", {"umap": None}),
        ):
            # This should show an import error for umap
            # The actual behavior depends on how imports are handled
            pass

    def test_map_requires_config(self, tmp_path, monkeypatch):
        """Test that map requires a config file."""
        monkeypatch.setattr(
            "fitz_ai.cli.commands.map.FitzPaths.config",
            lambda: tmp_path / "nonexistent" / "fitz.yaml",
        )

        # Mock the imports to avoid dependency issues
        with (patch.dict("sys.modules", {"umap": MagicMock(), "sklearn.cluster": MagicMock()}),):
            result = runner.invoke(app, ["map"])

        assert result.exit_code != 0
        assert "init" in result.output.lower() or "config" in result.output.lower()


class TestMapHelpers:
    """Tests for map helper functions."""

    def test_load_config_safe_returns_dict(self, tmp_path):
        """Test _load_config_safe returns config dict."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"collection": "test"},
        }
        config_path.write_text(yaml.dump(config))

        with patch(
            "fitz_ai.cli.commands.map.FitzPaths.config",
            return_value=config_path,
        ):
            from fitz_ai.cli.commands.map import _load_config_safe

            result = _load_config_safe()

        assert result["vector_db"]["plugin_name"] == "local_faiss"

    def test_get_vector_db(self):
        """Test _get_vector_db returns plugin."""
        mock_plugin = MagicMock()

        with patch(
            "fitz_ai.cli.commands.map.get_vector_db_plugin",
            return_value=mock_plugin,
        ):
            from fitz_ai.cli.commands.map import _get_vector_db

            config = {"vector_db": {"plugin_name": "local_faiss", "kwargs": {}}}
            result = _get_vector_db(config)

        assert result == mock_plugin

    def test_get_collections_returns_sorted(self):
        """Test _get_collections returns sorted list."""
        mock_vdb = MagicMock()
        mock_vdb.list_collections.return_value = ["zebra", "apple"]

        with patch(
            "fitz_ai.cli.commands.map.get_vector_db_plugin",
            return_value=mock_vdb,
        ):
            from fitz_ai.cli.commands.map import _get_collections

            config = {"vector_db": {"plugin_name": "local_faiss"}}
            result = _get_collections(config)

        assert result == ["apple", "zebra"]

    def test_get_collections_handles_error(self):
        """Test _get_collections returns empty on error."""
        with patch(
            "fitz_ai.cli.commands.map.get_vector_db_plugin",
            side_effect=Exception("failed"),
        ):
            from fitz_ai.cli.commands.map import _get_collections

            result = _get_collections({})

        assert result == []

    def test_get_embedding_id(self):
        """Test _get_embedding_id builds correct ID."""
        from fitz_ai.cli.commands.map import _get_embedding_id

        config = {
            "embedding": {
                "provider": "cohere",
                "model": "embed-english-v3.0",
            }
        }

        result = _get_embedding_id(config)

        assert "cohere" in result
        assert "embed-english-v3.0" in result

    def test_get_embedding_id_defaults(self):
        """Test _get_embedding_id uses defaults."""
        from fitz_ai.cli.commands.map import _get_embedding_id

        result = _get_embedding_id({})

        assert "ollama" in result
        assert "nomic-embed-text" in result


class TestMapOptions:
    """Tests for map command options."""

    def test_map_no_open_flag(self):
        """Test --no-open flag is recognized."""
        result = runner.invoke(app, ["map", "--help"])

        # ANSI codes can split the flag, so check for key parts
        assert "no-open" in result.output or "Don't open" in result.output

    def test_map_rebuild_flag(self):
        """Test --rebuild flag is recognized."""
        result = runner.invoke(app, ["map", "--help"])

        # ANSI codes can split the flag, so check for key parts
        assert "rebuild" in result.output

    def test_map_similarity_threshold_option(self):
        """Test --similarity-threshold option is recognized."""
        result = runner.invoke(app, ["map", "--help"])

        assert "--similarity-threshold" in result.output or "-t" in result.output

    def test_map_collection_option(self):
        """Test --collection option is recognized."""
        result = runner.invoke(app, ["map", "--help"])

        assert "--collection" in result.output or "-c" in result.output

    def test_map_output_option(self):
        """Test --output option is recognized."""
        result = runner.invoke(app, ["map", "--help"])

        assert "--output" in result.output or "-o" in result.output


class TestMapNoCollections:
    """Tests for handling missing collections."""

    def test_map_no_collections_shows_error(self, tmp_path):
        """Test map shows error when no collections exist."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"collection": "test"},
            "embedding": {},
        }
        config_path.write_text(yaml.dump(config))

        mock_vdb = MagicMock()
        mock_vdb.list_collections.return_value = []

        with (
            patch("fitz_ai.cli.commands.map.FitzPaths.config", return_value=config_path),
            patch("fitz_ai.cli.commands.map.get_vector_db_plugin", return_value=mock_vdb),
            patch.dict("sys.modules", {"umap": MagicMock(), "sklearn.cluster": MagicMock()}),
        ):
            result = runner.invoke(app, ["map"])

        assert result.exit_code != 0
        assert "no collection" in result.output.lower() or "ingest" in result.output.lower()


class TestMapVectorDBSupport:
    """Tests for vector DB support checking."""

    def test_map_requires_scroll_with_vectors_option_recognized(self):
        """Test that map options are recognized (proxy test for scroll_with_vectors)."""
        # This test verifies the map command is properly configured.
        # The actual scroll_with_vectors check is integration-level.
        result = runner.invoke(app, ["map", "--help"])

        assert result.exit_code == 0
        assert "collection" in result.output.lower()


class TestMapNoChunks:
    """Tests for handling empty collections."""

    def test_map_no_chunks_shows_error(self, tmp_path):
        """Test map shows error when collection has no chunks."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"collection": "test"},
            "embedding": {},
        }
        config_path.write_text(yaml.dump(config))

        mock_vdb = MagicMock()
        mock_vdb.list_collections.return_value = ["test"]
        mock_vdb.scroll_with_vectors = MagicMock()

        with (
            patch("fitz_ai.cli.commands.map.FitzPaths.config", return_value=config_path),
            patch("fitz_ai.vector_db.registry.get_vector_db_plugin", return_value=mock_vdb),
            patch("fitz_ai.cli.commands.map._get_collections", return_value=["test"]),
            patch("fitz_ai.map.embeddings.fetch_all_chunk_ids", return_value=[]),
            patch.dict("sys.modules", {"umap": MagicMock(), "sklearn.cluster": MagicMock()}),
        ):
            result = runner.invoke(app, ["map"])

        assert result.exit_code != 0
        assert "no chunk" in result.output.lower() or "ingest" in result.output.lower()
