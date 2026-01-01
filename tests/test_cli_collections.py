# tests/test_cli_collections.py
"""
Tests for the collections command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestCollectionsCommand:
    """Tests for fitz collections command."""

    def test_collections_shows_help(self):
        """Test that collections --help works."""
        result = runner.invoke(app, ["collections", "--help"])

        assert result.exit_code == 0
        assert "collection" in result.output.lower()

    def test_collections_requires_config(self, tmp_path, monkeypatch):
        """Test that collections requires a config file."""
        monkeypatch.setattr(
            "fitz_ai.cli.context.FitzPaths.config",
            lambda: tmp_path / "nonexistent" / "fitz.yaml",
        )

        result = runner.invoke(app, ["collections"], input="4\n")  # Exit option

        assert result.exit_code != 0
        assert "init" in result.output.lower() or "config" in result.output.lower()


class TestCollectionsHelpers:
    """Tests for collections helper functions."""

    def test_cli_context_loads_config(self, tmp_path):
        """Test CLIContext loads config."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"vector_db": {"plugin_name": "local_faiss"}}
        config_path.write_text(yaml.dump(config))

        with patch(
            "fitz_ai.cli.context.FitzPaths.engine_config",
            return_value=config_path,
        ):
            from fitz_ai.cli.context import CLIContext

            ctx = CLIContext.load_or_none()

        assert ctx is not None
        assert ctx.vector_db_plugin == "local_faiss"

    def test_get_vector_client(self):
        """Test _get_vector_client returns plugin."""
        mock_plugin = MagicMock()

        with patch(
            "fitz_ai.vector_db.registry.get_vector_db_plugin",
            return_value=mock_plugin,
        ):
            from fitz_ai.cli.commands.collections import _get_vector_client

            client = _get_vector_client("local_faiss", {})

        assert client == mock_plugin

    def test_get_available_vector_dbs(self, tmp_path):
        """Test _get_available_vector_dbs returns sorted list."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"vector_db": {"plugin_name": "local_faiss", "kwargs": {}}}
        config_path.write_text(yaml.dump(config))

        with (
            patch("fitz_ai.cli.context.FitzPaths.engine_config", return_value=config_path),
            patch(
                "fitz_ai.vector_db.registry.available_vector_db_plugins",
                return_value=["qdrant", "local_faiss"],
            ),
        ):
            from fitz_ai.cli.commands.collections import _get_available_vector_dbs
            from fitz_ai.cli.context import CLIContext

            ctx = CLIContext.load()
            result = _get_available_vector_dbs(ctx)

        # Configured one should be first
        assert result[0]["name"] == "local_faiss"
        assert result[0]["is_configured"] is True


class TestDisplayCollectionsTable:
    """Tests for _display_collections_table."""

    def test_display_collections_table_plain(self, capsys):
        """Test _display_collections_table outputs table."""
        with patch("fitz_ai.cli.commands.collections.RICH", False):
            from fitz_ai.cli.commands.collections import _display_collections_table

            collections = [
                {"name": "docs", "count": 100, "status": "ready"},
                {"name": "code", "count": 50, "status": "ready"},
            ]

            _display_collections_table(collections)

        captured = capsys.readouterr()
        assert "docs" in captured.out
        assert "100" in captured.out
        assert "code" in captured.out
        assert "50" in captured.out


class TestDisplayCollectionInfo:
    """Tests for _display_collection_info."""

    def test_display_collection_info_plain(self, capsys):
        """Test _display_collection_info shows stats."""
        with patch("fitz_ai.cli.commands.collections.RICH", False):
            from fitz_ai.cli.commands.collections import _display_collection_info

            stats = {
                "points_count": 150,
                "vector_size": 768,
                "status": "ready",
            }

            _display_collection_info("my_collection", stats)

        captured = capsys.readouterr()
        assert "my_collection" in captured.out
        assert "150" in captured.out
        assert "768" in captured.out

    def test_display_collection_info_vectors_count(self, capsys):
        """Test _display_collection_info handles vectors_count key."""
        with patch("fitz_ai.cli.commands.collections.RICH", False):
            from fitz_ai.cli.commands.collections import _display_collection_info

            stats = {"vectors_count": 200}

            _display_collection_info("test", stats)

        captured = capsys.readouterr()
        assert "200" in captured.out


class TestCollectionsNoCollections:
    """Tests for handling empty collections."""

    def test_collections_empty_shows_message(self, tmp_path):
        """Test collections shows helpful message when no collections."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"vector_db": {"plugin_name": "local_faiss"}}
        config_path.write_text(yaml.dump(config))

        mock_client = MagicMock()
        mock_client.list_collections.return_value = []

        with (
            patch(
                "fitz_ai.cli.context.FitzPaths.config",
                return_value=config_path,
            ),
            patch(
                "fitz_ai.vector_db.registry.get_vector_db_plugin",
                return_value=mock_client,
            ),
            patch(
                "fitz_ai.vector_db.registry.available_vector_db_plugins",
                return_value=["local_faiss"],
            ),
        ):
            result = runner.invoke(app, ["collections"])

        assert "no collection" in result.output.lower() or "ingest" in result.output.lower()


class TestCollectionsWithData:
    """Tests for collections with actual data."""

    def test_collections_lists_and_exits(self, tmp_path):
        """Test collections can list and exit."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"vector_db": {"plugin_name": "local_faiss"}}
        config_path.write_text(yaml.dump(config))

        mock_client = MagicMock()
        mock_client.list_collections.return_value = ["docs", "code"]
        mock_client.get_collection_stats.return_value = {
            "points_count": 100,
            "status": "ready",
        }

        with (
            patch(
                "fitz_ai.cli.context.FitzPaths.config",
                return_value=config_path,
            ),
            patch(
                "fitz_ai.vector_db.registry.get_vector_db_plugin",
                return_value=mock_client,
            ),
            patch(
                "fitz_ai.vector_db.registry.available_vector_db_plugins",
                return_value=["local_faiss"],
            ),
        ):
            # Select "Exit" (option 3 - after docs, code)
            result = runner.invoke(app, ["collections"], input="3\n")

        assert "docs" in result.output
        assert "code" in result.output

    def test_collections_show_example_chunks(self, tmp_path):
        """Test collections can show example chunks."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"vector_db": {"plugin_name": "local_faiss"}}
        config_path.write_text(yaml.dump(config))

        mock_record = MagicMock()
        mock_record.payload = {"content": "Sample chunk content", "doc_id": "doc1"}
        mock_record.id = "chunk_1"

        mock_client = MagicMock()
        mock_client.list_collections.return_value = ["docs"]
        mock_client.get_collection_stats.return_value = {
            "points_count": 10,
            "status": "ready",
        }
        mock_client.scroll.return_value = ([mock_record], None)

        with (
            patch(
                "fitz_ai.cli.context.FitzPaths.config",
                return_value=config_path,
            ),
            patch(
                "fitz_ai.vector_db.registry.get_vector_db_plugin",
                return_value=mock_client,
            ),
            patch(
                "fitz_ai.vector_db.registry.available_vector_db_plugins",
                return_value=["local_faiss"],
            ),
            patch("fitz_ai.cli.commands.collections.RICH", False),
        ):
            # Select first collection, show chunks, then exit
            result = runner.invoke(app, ["collections"], input="1\n1\n\n4\n")

        assert "Sample chunk" in result.output or mock_client.scroll.called


class TestCollectionsDelete:
    """Tests for collection deletion."""

    def test_collections_delete_cancelled(self, tmp_path):
        """Test delete can be cancelled."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"vector_db": {"plugin_name": "local_faiss"}}
        config_path.write_text(yaml.dump(config))

        mock_client = MagicMock()
        mock_client.list_collections.return_value = ["docs"]
        mock_client.get_collection_stats.return_value = {"points_count": 10}

        with (
            patch(
                "fitz_ai.cli.context.FitzPaths.config",
                return_value=config_path,
            ),
            patch(
                "fitz_ai.vector_db.registry.get_vector_db_plugin",
                return_value=mock_client,
            ),
            patch(
                "fitz_ai.vector_db.registry.available_vector_db_plugins",
                return_value=["local_faiss"],
            ),
        ):
            # Select collection, select delete, answer no, exit
            result = runner.invoke(app, ["collections"], input="1\n2\nn\n4\n")

        # Delete should not be called
        mock_client.delete_collection.assert_not_called()
        assert "cancel" in result.output.lower()
