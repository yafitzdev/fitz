import pytest
from typer.testing import CliRunner

from fitz_rag.cli import app, CLIError

runner = CliRunner()


# ---------------------------------------------------------
# Smoke test: CLI loads
# ---------------------------------------------------------
def test_cli_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Fitz-RAG" in result.stdout


# ---------------------------------------------------------
# Test top-level commands using --help
# ---------------------------------------------------------
def test_all_top_level_commands_help():
    # app.registered_commands is a list of CommandInfo
    for cmd in app.registered_commands:
        name = cmd.name

        # Skip hidden/internal Typer commands
        if name is None:
            continue

        # Skip group commands (collections)
        if name == "collections":
            continue

        result = runner.invoke(app, [name, "--help"])
        assert result.exit_code == 0, f"Help failed for '{name}'"


# ---------------------------------------------------------
# Test version
# ---------------------------------------------------------
def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "fitz-rag version" in result.stdout


# ---------------------------------------------------------
# Test config-path
# ---------------------------------------------------------
def test_cli_config_path():
    result = runner.invoke(app, ["config-path"])
    assert result.exit_code == 0


# ---------------------------------------------------------
# Test config-show with correct monkeypatch target
# ---------------------------------------------------------
def test_cli_config_show(monkeypatch):

    # Patch the *actual import* inside cli.py
    monkeypatch.setattr(
        "fitz_rag.cli.get_config",
        lambda: {"test": True}
    )

    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    assert "test" in result.stdout


# ---------------------------------------------------------
# collections list — with correct monkeypatch
# ---------------------------------------------------------
def test_cli_collections_list(monkeypatch):

    class MockQdrant:
        class _Cols:
            collections = [type("C", (), {"name": "test_collection"})]
        def get_collections(self):
            return self._Cols()

    # Patch the exact function used inside cli.py
    monkeypatch.setattr(
        "fitz_rag.cli.create_qdrant_client",
        lambda: MockQdrant()
    )

    result = runner.invoke(app, ["collections", "list"])
    assert result.exit_code == 0
    assert "test_collection" in result.stdout


# ---------------------------------------------------------
# collections drop
# ---------------------------------------------------------
def test_cli_collections_drop(monkeypatch):

    class MockQdrant:
        def delete_collection(self, name):
            pass

    monkeypatch.setattr(
        "fitz_rag.cli.create_qdrant_client",
        lambda: MockQdrant()
    )

    result = runner.invoke(app, ["collections", "drop", "demo"], input="y\n")
    assert result.exit_code == 0
    assert "Deleted collection: demo" in result.stdout


# ---------------------------------------------------------
# query help only
# ---------------------------------------------------------
def test_cli_query_help():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "Run a full retrieval" in result.stdout
