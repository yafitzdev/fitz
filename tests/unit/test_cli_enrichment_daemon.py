"""Tests for the hidden enrichment daemon command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_sage.cli.commands.enrichment_daemon import _remove_owned_pid_file
from fitz_sage.cli.cli import app

runner = CliRunner()


def test_enrichment_daemon_loads_collection_and_continues_enrichment() -> None:
    """Hidden daemon command resumes enrichment for the persisted collection."""
    mock_engine = MagicMock()
    mock_registry = MagicMock()
    mock_registry.list.return_value = ["fitz_krag"]

    with (
        patch(
            "fitz_sage.cli.commands.enrichment_daemon.get_engine_registry",
            return_value=mock_registry,
        ),
        patch(
            "fitz_sage.cli.commands.enrichment_daemon.get_default_engine",
            return_value="fitz_krag",
        ),
        patch("fitz_sage.cli.commands.enrichment_daemon.create_engine", return_value=mock_engine),
    ):
        result = runner.invoke(
            app,
            ["enrichment-daemon", "--collection", "docs"],
        )

    assert result.exit_code == 0
    mock_engine.load.assert_called_once_with("docs")
    mock_engine.continue_enrichment.assert_called_once()


def test_daemon_only_removes_its_own_pid_file(monkeypatch, tmp_path) -> None:
    pid_path = tmp_path / "enrichment_daemon.pid"
    monkeypatch.setattr("fitz_sage.cli.commands.enrichment_daemon.os.getpid", lambda: 123)

    pid_path.write_text("456", encoding="utf-8")
    _remove_owned_pid_file(pid_path)
    assert pid_path.exists()

    pid_path.write_text("123", encoding="utf-8")
    _remove_owned_pid_file(pid_path)
    assert not pid_path.exists()
