# tests/test_cli_ingest_smoke.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from typer.testing import CliRunner

import ingest.cli as ingest_cli


class DummyEmbeddingPlugin:
    def embed(self, text: str) -> list[float]:
        return [0.0, 1.0, 2.0]


class DummyVectorClient:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def upsert(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def test_ingest_cli_smoke(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    # patch CLI-imported symbols (NOT the original modules)
    def _fake_get_llm_plugin(*, plugin_name: str, plugin_type: Any) -> Type[Any]:
        return DummyEmbeddingPlugin

    def _fake_get_vector_db_plugin(plugin_name: str, *args: Any, **kwargs: Any) -> Type[Any]:
        return DummyVectorClient

    monkeypatch.setattr(ingest_cli, "get_llm_plugin", _fake_get_llm_plugin)
    monkeypatch.setattr(ingest_cli, "get_vector_db_plugin", _fake_get_vector_db_plugin)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("a.txt").write_text("hello world", encoding="utf-8")

        # IMPORTANT: ingest_cli.app is a *single-command* CLI (root command is "run"),
        # so you do NOT include "run" in argv.
        result = runner.invoke(
            ingest_cli.app,
            [
                "a.txt",
                "--collection",
                "test_collection",
                "--ingest-plugin",
                "local",
                "--embedding-plugin",
                "whatever",
                "--vector-db-plugin",
                "whatever",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "OK:" in result.stdout
