# tests/unit/test_cli_query.py
"""
Tests for the query command.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from fitz_sage.cli.cli import app
from fitz_sage.core import EvidencePack
from fitz_sage.core.answer_mode import AnswerMode

runner = CliRunner()


@pytest.fixture(autouse=True)
def _skip_firstrun():
    """Skip first-run detection in all query CLI tests."""
    with patch("fitz_sage.config.firstrun.needs_firstrun", return_value=False):
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

    def test_query_defaults_to_current_directory(self, tmp_path):
        """Query should register the current directory when no flags are supplied."""
        pack = EvidencePack(query="test question", mode=AnswerMode.SUFFICIENT)
        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_caps.supports_collections = True
        mock_registry.get_capabilities.return_value = mock_caps

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry",
                return_value=mock_registry,
            ),
            patch(
                "fitz_sage.cli.commands.retrieve.get_default_engine",
                return_value="fitz_krag",
            ),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.Path.cwd", return_value=tmp_path),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack") as mock_display,
        ):
            result = runner.invoke(app, ["query", "test question"])

        assert result.exit_code == 0
        mock_engine.point.assert_called_once()
        assert mock_engine.point.call_args.args[0] == tmp_path
        assert mock_engine.point.call_args.args[1] == tmp_path.name
        mock_engine.wait_for_query_surface.assert_called_once()
        mock_engine.evidence.assert_called_once()
        mock_display.assert_called_once_with(pack, max_items=10)


class TestQueryExecution:
    """Tests for query execution with mocked engine (persistent ingest path)."""

    def test_query_direct_mode(self):
        """Test query with direct question argument via retrieval path."""
        pack = EvidencePack(query="What is RAG?", mode=AnswerMode.SUFFICIENT)
        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry",
                return_value=mock_registry,
            ),
            patch(
                "fitz_sage.cli.commands.retrieve.get_default_engine",
                return_value="fitz_krag",
            ),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.Path.cwd", return_value=Path("docs")),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack"),
        ):
            result = runner.invoke(app, ["query", "What is RAG?"])

        mock_engine.evidence.assert_called_once()
        assert result.exit_code == 0

    def test_query_handles_error(self):
        """Test query handles errors gracefully."""
        mock_engine = MagicMock()
        mock_engine.evidence.side_effect = Exception("Test error")

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry",
                return_value=mock_registry,
            ),
            patch(
                "fitz_sage.cli.commands.retrieve.get_default_engine",
                return_value="fitz_krag",
            ),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.Path.cwd", return_value=Path("docs")),
        ):
            result = runner.invoke(app, ["query", "What is RAG?"])

        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "error" in result.output.lower()


class TestQueryOptions:
    """Tests for query command options."""

    def test_query_with_collection_option(self):
        """Test query with --collection option."""
        pack = EvidencePack(query="question", mode=AnswerMode.SUFFICIENT)
        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.requires_documents_at_query = False
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["custom"]

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry",
                return_value=mock_registry,
            ),
            patch(
                "fitz_sage.cli.commands.retrieve.get_default_engine",
                return_value="fitz_krag",
            ),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack"),
        ):
            runner.invoke(app, ["query", "question", "-c", "custom"])

        mock_engine.load.assert_called_once_with("custom")
        mock_engine.evidence.assert_called_once()

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
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry",
                return_value=mock_registry,
            ),
            patch(
                "fitz_sage.cli.commands.retrieve.get_default_engine",
                return_value="fitz_krag",
            ),
        ):
            result = runner.invoke(app, ["query", "question", "-c", "nonexistent"])

        assert "not found" in result.output.lower() or "available" in result.output.lower()
