# tests/unit/test_cli_query.py
"""
Tests for the query command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from fitz_sage.cli.cli import app
from fitz_sage.core import Answer

runner = CliRunner()


@pytest.fixture(autouse=True)
def _skip_firstrun():
    """Skip first-run detection in all query CLI tests."""
    with patch("fitz_sage.core.firstrun.needs_firstrun", return_value=False):
        yield


class TestQueryCommand:
    """Tests for fitz query command."""

    def test_query_shows_help(self):
        """Test that query --help works."""
        result = runner.invoke(app, ["query", "--help"])

        assert result.exit_code == 0
        assert "Query your knowledge base" in result.output or "query" in result.output.lower()

    def test_answer_shows_help(self):
        """Test that answer --help works."""
        result = runner.invoke(app, ["answer", "--help"])

        assert result.exit_code == 0
        assert "synthesis" in result.output.lower() or "answer" in result.output.lower()

    def test_query_no_collections_exits(self):
        """Test that query exits when no collections found."""
        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_caps.supports_collections = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = []

        with (
            patch("fitz_sage.cli.commands.query.get_engine_registry", return_value=mock_registry),
            patch("fitz_sage.cli.commands.query.get_default_engine", return_value="fitz_krag"),
        ):
            result = runner.invoke(app, ["query", "test question"])

        assert "no" in result.output.lower() or "ingest" in result.output.lower()


class TestQueryHelpers:
    """Tests for query helper functions."""

    def test_cli_context_loads_config(self, tmp_path):
        """Test CLIContext.load() returns raw and typed config."""
        import yaml

        from fitz_sage.cli.context import CLIContext

        # Create flat config file
        config_path = tmp_path / "config.yaml"
        config = {
            "chat_fast": "endpoint/qwen2.5-7b-instruct",
            "chat_balanced": "endpoint/qwen2.5-7b-instruct",
            "chat_smart": "openai/gpt-4o",
            "chat_base_url": None,
            "collection": "test",
        }
        config_path.write_text(yaml.dump(config))

        with patch(
            "fitz_sage.cli.context.FitzPaths.config",
            return_value=config_path,
        ):
            ctx = CLIContext.load(engine="fitz_krag")

        assert ctx.raw_config["chat_smart"] == "openai/gpt-4o"
        assert ctx.typed_config.collection == "test"

    def test_get_collections_returns_list(self):
        """Test get_collections returns collection list."""
        mock_ctx = MagicMock()
        mock_ctx.get_collections.return_value = ["coll_a", "coll_b"]

        from fitz_sage.cli.utils import get_collections

        collections = get_collections(mock_ctx)

        assert sorted(collections) == ["coll_a", "coll_b"]

    def test_get_collections_handles_empty(self):
        """Test get_collections returns empty list when no collections."""
        mock_ctx = MagicMock()
        mock_ctx.get_collections.return_value = []

        from fitz_sage.cli.utils import get_collections

        collections = get_collections(mock_ctx)

        assert collections == []


class TestQueryExecution:
    """Tests for query execution with mocked engine (persistent ingest path)."""

    def test_query_direct_mode(self):
        """Test query with direct question argument via persistent ingest path."""
        mock_answer = Answer(
            text="This is the answer",
            provenance=[],
            mode="trustworthy",
        )

        mock_engine = MagicMock()
        mock_engine.answer.return_value = mock_answer

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch("fitz_sage.cli.commands.query.get_engine_registry", return_value=mock_registry),
            patch("fitz_sage.cli.commands.query.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.query.create_engine", return_value=mock_engine),
        ):
            result = runner.invoke(app, ["query", "What is RAG?"])

        mock_engine.answer.assert_called_once()
        assert result.exit_code == 0

    def test_query_handles_error(self):
        """Test query handles errors gracefully."""
        mock_engine = MagicMock()
        mock_engine.answer.side_effect = Exception("Test error")

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch("fitz_sage.cli.commands.query.get_engine_registry", return_value=mock_registry),
            patch("fitz_sage.cli.commands.query.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.query.create_engine", return_value=mock_engine),
        ):
            result = runner.invoke(app, ["query", "What is RAG?"])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


class TestQueryOptions:
    """Tests for query command options."""

    def test_query_with_collection_option(self):
        """Test query with --collection option."""
        mock_answer = Answer(
            text="Answer",
            provenance=[],
            mode="trustworthy",
        )

        mock_engine = MagicMock()
        mock_engine.answer.return_value = mock_answer

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["custom"]

        with (
            patch("fitz_sage.cli.commands.query.get_engine_registry", return_value=mock_registry),
            patch("fitz_sage.cli.commands.query.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.query.create_engine", return_value=mock_engine),
        ):
            runner.invoke(app, ["query", "question", "-c", "custom"])

        mock_engine.load.assert_called_once_with("custom")
        mock_engine.answer.assert_called_once()

    def test_query_collection_not_found(self):
        """Test query shows error when collection not found."""
        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["other"]

        with (
            patch("fitz_sage.cli.commands.query.get_engine_registry", return_value=mock_registry),
            patch("fitz_sage.cli.commands.query.get_default_engine", return_value="fitz_krag"),
        ):
            result = runner.invoke(app, ["query", "question", "-c", "nonexistent"])

        assert "not found" in result.output.lower() or "available" in result.output.lower()
