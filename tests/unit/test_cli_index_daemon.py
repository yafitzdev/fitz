# tests/unit/test_cli_index_daemon.py
"""Tests for the hidden indexing daemon command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_sage.cli.cli import app

runner = CliRunner()


def test_index_daemon_loads_collection_and_continues_indexing() -> None:
    """Hidden daemon command resumes the persisted collection."""
    mock_engine = MagicMock()
    mock_registry = MagicMock()
    mock_registry.list.return_value = ["fitz_krag"]

    with (
        patch(
            "fitz_sage.cli.commands.index_daemon.get_engine_registry",
            return_value=mock_registry,
        ),
        patch(
            "fitz_sage.cli.commands.index_daemon.get_default_engine",
            return_value="fitz_krag",
        ),
        patch("fitz_sage.cli.commands.index_daemon.create_engine", return_value=mock_engine),
    ):
        result = runner.invoke(
            app,
            ["index-daemon", "--collection", "docs"],
        )

    assert result.exit_code == 0
    mock_engine.load.assert_called_once_with("docs")
    mock_engine.continue_indexing.assert_called_once()
