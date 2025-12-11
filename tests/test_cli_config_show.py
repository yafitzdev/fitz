from typer.testing import CliRunner
from fitz_rag.cli import app

runner = CliRunner()

def test_cli_config_show_runs():
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    assert "retriever" in result.stdout.lower()
