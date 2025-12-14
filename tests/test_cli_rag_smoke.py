from typer.testing import CliRunner

from fitz.pipeline.cli import app


def test_rag_cli_config_show_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0, result.output
    assert "embedding" in result.stdout
