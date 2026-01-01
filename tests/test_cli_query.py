# tests/test_cli_query.py
"""
Tests for the query command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestQueryCommand:
    """Tests for fitz query command."""

    def test_query_shows_help(self):
        """Test that query --help works."""
        result = runner.invoke(app, ["query", "--help"])

        assert result.exit_code == 0
        assert "Query your knowledge base" in result.output or "query" in result.output.lower()

    def test_query_requires_config(self, tmp_path, monkeypatch):
        """Test that query requires a config file."""
        # Set workspace to temp dir without config
        monkeypatch.setattr(
            "fitz_ai.core.paths.FitzPaths.config",
            lambda: tmp_path / "nonexistent" / "fitz.yaml",
        )
        monkeypatch.setattr(
            "fitz_ai.cli.context.FitzPaths.engine_config",
            lambda name: tmp_path / "nonexistent" / f"{name}.yaml",
        )

        result = runner.invoke(app, ["query", "test question", "--engine", "fitz_rag"])

        assert result.exit_code != 0
        assert "init" in result.output.lower() or "config" in result.output.lower()


class TestQueryHelpers:
    """Tests for query helper functions."""

    def test_load_fitz_rag_config_returns_config(self, tmp_path):
        """Test load_fitz_rag_config loads valid config."""
        import yaml

        # Create engine-specific config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_path = config_dir / "fitz_rag.yaml"
        config = {
            "chat": {"plugin_name": "cohere"},
            "embedding": {"plugin_name": "cohere"},
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"plugin_name": "dense", "collection": "test"},
        }
        config_path.write_text(yaml.dump(config))

        with patch(
            "fitz_ai.cli.context.FitzPaths.engine_config",
            return_value=config_path,
        ):
            from fitz_ai.cli.utils import load_fitz_rag_config

            raw, typed = load_fitz_rag_config()

        assert raw["chat"]["plugin_name"] == "cohere"
        assert typed.retrieval.collection == "test"

    def test_get_collections_returns_list(self):
        """Test get_collections returns collection list."""
        mock_vdb = MagicMock()
        mock_vdb.list_collections.return_value = ["coll_a", "coll_b"]

        with patch(
            "fitz_ai.vector_db.registry.get_vector_db_plugin",
            return_value=mock_vdb,
        ):
            from fitz_ai.cli.utils import get_collections

            collections = get_collections({"vector_db": {"plugin_name": "local_faiss"}})

        assert sorted(collections) == ["coll_a", "coll_b"]

    def test_get_collections_handles_error(self):
        """Test get_collections returns empty list on error."""
        with patch(
            "fitz_ai.vector_db.registry.get_vector_db_plugin",
            side_effect=Exception("connection failed"),
        ):
            from fitz_ai.cli.utils import get_collections

            collections = get_collections({})

        assert collections == []


class TestQueryExecution:
    """Tests for query execution with mocked engine."""

    def test_query_direct_mode(self, tmp_path):
        """Test query with direct question argument."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "chat": {"plugin_name": "cohere", "kwargs": {"model": "command"}},
            "embedding": {
                "plugin_name": "cohere",
                "kwargs": {"model": "embed-english-v3.0"},
            },
            "vector_db": {"plugin_name": "local_faiss", "kwargs": {}},
            "retrieval": {"plugin_name": "dense", "collection": "test", "top_k": 5},
            "rerank": {"enabled": False},
        }
        config_path.write_text(yaml.dump(config))

        mock_answer = MagicMock()
        mock_answer.text = "This is the answer"
        mock_answer.provenance = []

        mock_engine = MagicMock()
        mock_engine.answer.return_value = mock_answer

        mock_vdb = MagicMock()
        mock_vdb.list_collections.return_value = ["test"]

        with (
            patch("fitz_ai.cli.context.FitzPaths.engine_config", return_value=config_path),
            patch("fitz_ai.cli.context.FitzPaths.config", return_value=config_path),
            patch("fitz_ai.cli.commands.query.create_engine", return_value=mock_engine),
            patch("fitz_ai.vector_db.registry.get_vector_db_plugin", return_value=mock_vdb),
        ):
            runner.invoke(app, ["query", "What is RAG?", "--engine", "fitz_rag"])

        # Should call engine.answer with the question
        mock_engine.answer.assert_called_once()


class TestQueryOptions:
    """Tests for query command options."""

    def test_query_with_collection_option(self, tmp_path):
        """Test query with --collection option."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "chat": {"plugin_name": "cohere"},
            "embedding": {"plugin_name": "cohere"},
            "vector_db": {"plugin_name": "local_faiss"},
            "retrieval": {"plugin_name": "dense", "collection": "default"},
        }
        config_path.write_text(yaml.dump(config))

        mock_answer = MagicMock()
        mock_answer.text = "Answer"
        mock_answer.provenance = []

        mock_engine = MagicMock()
        mock_engine.answer.return_value = mock_answer

        mock_vdb = MagicMock()
        mock_vdb.list_collections.return_value = ["custom"]

        with (
            patch("fitz_ai.cli.context.FitzPaths.engine_config", return_value=config_path),
            patch("fitz_ai.cli.context.FitzPaths.config", return_value=config_path),
            patch("fitz_ai.cli.commands.query.create_engine", return_value=mock_engine),
            patch("fitz_ai.vector_db.registry.get_vector_db_plugin", return_value=mock_vdb),
        ):
            runner.invoke(app, ["query", "question", "-c", "custom", "--engine", "fitz_rag"])

        # Engine should be created (we can't easily verify the collection was set)
        assert mock_engine.answer.called
