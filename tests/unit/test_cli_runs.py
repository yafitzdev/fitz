"""Tests for retrieval-run inspection and replay commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_sage.cli.cli import app
from fitz_sage.core import (
    EvidencePack,
    GovernanceExecution,
    QueryExecution,
    RetrievalRun,
    RunEnvironment,
)
from fitz_sage.core.answer_mode import AnswerMode

runner = CliRunner()


def _write_trace(path):
    run = RetrievalRun(
        run_id="run-1",
        created_at="2026-07-26T10:00:00Z",
        query=QueryExecution(
            source_text="What changed?",
            sanitized_text="What changed?",
            retrieval_text="changed",
            query_shape="narrow",
        ),
        evidence=EvidencePack(
            query="What changed?",
            mode=AnswerMode.INSUFFICIENT,
        ),
        strategies=(),
        candidate_stages=(),
        governance=GovernanceExecution(
            mode="insufficient",
            evaluated=0,
            selected=0,
            max_documents=0,
            query_shape="narrow",
            minimum_sufficient_documents=0,
        ),
        ranked_evidence=(),
        environment=RunEnvironment(
            fitz_sage_version="0.15.0",
            engine="fitz_krag",
            collection="docs",
            config_sha256="sha",
            collection_sha256=None,
        ),
    )
    run.write(path)


def test_explain_prints_recorded_execution_without_rerunning(tmp_path):
    trace_path = tmp_path / "run.json"
    _write_trace(trace_path)

    result = runner.invoke(app, ["explain", str(trace_path)])

    assert result.exit_code == 0
    assert "Retrieval run run-1" in result.output
    assert "Query: What changed?" in result.output
    assert "Replay: unavailable" in result.output


def test_replay_writes_result_and_preserves_json_stdout(tmp_path):
    trace_path = tmp_path / "run.json"
    output_path = tmp_path / "replay.json"
    replay = MagicMock()
    replay.to_json.return_value = '{"changed": false}'

    with patch(
        "fitz_sage.cli.commands.runs.replay_governance",
        return_value=replay,
    ) as replay_call:
        result = runner.invoke(
            app,
            [
                "replay",
                str(trace_path),
                "--governance",
                "pyrrho/test",
                "--output",
                str(output_path),
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0
    assert result.output.strip() == '{"changed": false}'
    replay_call.assert_called_once_with(trace_path, "pyrrho/test")
    replay.write.assert_called_once_with(output_path, include_content=False)
    replay.to_json.assert_called_once_with(include_content=False)


def test_explain_reports_invalid_trace(tmp_path):
    trace_path = tmp_path / "invalid.json"
    trace_path.write_text("not-json", encoding="utf-8")

    result = runner.invoke(app, ["explain", str(trace_path)])

    assert result.exit_code == 1
    assert "Explain failed" in result.output
