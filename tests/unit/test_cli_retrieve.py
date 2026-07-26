# tests/unit/test_cli_retrieve.py
"""Tests for the retrieval-first CLI command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_sage.cli.cli import app
from fitz_sage.core import EvidenceItem, EvidencePack
from fitz_sage.core.answer_mode import AnswerMode

runner = CliRunner()


class TestRetrieveCommand:
    """Tests for fitz retrieve."""

    def test_retrieve_text_displays_evidence_pack(self):
        """Text mode should render the EvidencePack through the CLI display helper."""
        pack = EvidencePack(
            query="Which test failed?",
            mode=AnswerMode.SUFFICIENT,
            items=[
                EvidenceItem(
                    rank=1,
                    source_id="doc-1",
                    file_path="docs/sprint.md",
                    address_kind="section",
                    address_location="Sprint 47",
                    line_range=(10, 12),
                    score=0.9,
                    excerpt="Payment retry failed.",
                    content="Payment retry failed.",
                )
            ],
            reasons=["Sources support the answer."],
            indexing_status={"complete": True},
        )

        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry", return_value=mock_registry
            ),
            patch("fitz_sage.cli.commands.retrieve.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack") as mock_display,
        ):
            result = runner.invoke(
                app,
                [
                    "retrieve",
                    "Which test failed?",
                    "--collection",
                    "test",
                    "--top-k",
                    "3",
                ],
            )

        assert result.exit_code == 0
        mock_engine.load.assert_called_once_with("test")
        mock_engine.evidence.assert_called_once()
        mock_display.assert_called_once_with(pack, max_items=3)

    def test_retrieve_json_outputs_evidence_pack_only(self):
        """JSON mode should emit a loadable EvidencePack without progress text."""
        pack = EvidencePack(
            query="Which test failed?",
            mode=AnswerMode.SUFFICIENT,
            items=[
                EvidenceItem(
                    rank=1,
                    source_id="doc-1",
                    file_path="docs/sprint.md",
                    address_kind="section",
                    address_location="Sprint 47",
                    line_range=(10, 12),
                    score=0.9,
                    excerpt="Payment retry failed.",
                    content="Payment retry failed.",
                )
            ],
            reasons=["Sources support the answer."],
            indexing_status={"complete": True},
        )

        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["test"]

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry", return_value=mock_registry
            ),
            patch("fitz_sage.cli.commands.retrieve.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
        ):
            result = runner.invoke(
                app,
                ["retrieve", "Which test failed?", "--collection", "test", "--format", "json"],
            )

        data = json.loads(result.output)

        assert result.exit_code == 0
        assert data["query"] == "Which test failed?"
        assert data["mode"] == "sufficient"
        mock_engine.load.assert_called_once_with("test")
        mock_engine.evidence.assert_called_once()

    def test_retrieve_trace_uses_same_execution_and_writes_requested_content(self, tmp_path):
        """Tracing replaces evidence(), so retrieval is executed exactly once."""
        pack = EvidencePack(
            query="Which test failed?",
            mode=AnswerMode.SUFFICIENT,
            indexing_status={"complete": True},
        )
        trace_path = tmp_path / "retrieval-run.json"
        run = MagicMock()
        run.evidence = pack
        run.write.return_value = trace_path.resolve()
        mock_engine = MagicMock()
        mock_engine.trace.return_value = run

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
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
            patch(
                "fitz_sage.cli.commands.retrieve.create_engine",
                return_value=mock_engine,
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "retrieve",
                    "Which test failed?",
                    "--collection",
                    "test",
                    "--format",
                    "json",
                    "--trace",
                    str(trace_path),
                    "--trace-content",
                ],
            )

        assert result.exit_code == 0
        assert json.loads(result.output)["mode"] == "sufficient"
        mock_engine.trace.assert_called_once()
        mock_engine.evidence.assert_not_called()
        run.write.assert_called_once_with(trace_path, include_content=True)

    def test_trace_content_requires_trace_path(self):
        result = runner.invoke(
            app,
            ["retrieve", "Which test failed?", "--trace-content"],
        )

        assert result.exit_code == 1
        assert "--trace-content requires --trace PATH" in result.output

    def test_retrieve_with_source_waits_for_required_indexing(self, tmp_path):
        """Source-backed retrieval waits for the query surface before evidence retrieval."""
        source = tmp_path / "docs"
        source.mkdir()

        pack = EvidencePack(query="What changed?", mode=AnswerMode.SUFFICIENT)
        manifest = MagicMock()
        manifest.entries.return_value = {}

        mock_engine = MagicMock()
        mock_engine.point.return_value = manifest
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry", return_value=mock_registry
            ),
            patch("fitz_sage.cli.commands.retrieve.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack"),
        ):
            result = runner.invoke(
                app,
                ["retrieve", "What changed?", "--source", str(source)],
            )

        assert result.exit_code == 0
        assert [method[0] for method in mock_engine.method_calls[:3]] == [
            "point",
            "wait_for_query_surface",
            "evidence",
        ]
        mock_engine.point.assert_called_once()
        assert mock_engine.point.call_args.args[0] == source
        assert mock_engine.point.call_args.args[1] == "docs"
        mock_engine.wait_for_query_surface.assert_called_once()
        mock_engine.evidence.assert_called_once()

    def test_retrieve_defaults_to_current_directory(self, tmp_path):
        """Retrieve should register the current directory when no flags are supplied."""
        pack = EvidencePack(query="What changed?", mode=AnswerMode.SUFFICIENT)

        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry", return_value=mock_registry
            ),
            patch("fitz_sage.cli.commands.retrieve.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.Path.cwd", return_value=tmp_path),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack"),
        ):
            result = runner.invoke(app, ["retrieve", "What changed?"])

        assert result.exit_code == 0
        mock_engine.point.assert_called_once()
        assert mock_engine.point.call_args.args[0] == tmp_path
        assert mock_engine.point.call_args.args[1] == tmp_path.name
        mock_engine.wait_for_query_surface.assert_called_once()
        mock_engine.evidence.assert_called_once()

    def test_retrieve_reuses_current_directory_collection_when_source_matches(self, tmp_path):
        """Repeated no-flag queries should not rebuild the same current-dir collection."""
        collection_dir = tmp_path / ".fitz" / "collections" / tmp_path.name
        collection_dir.mkdir(parents=True)
        (collection_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (collection_dir / "source_dir.txt").write_text(str(tmp_path), encoding="utf-8")

        pack = EvidencePack(query="What changed?", mode=AnswerMode.SUFFICIENT)

        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry", return_value=mock_registry
            ),
            patch("fitz_sage.cli.commands.retrieve.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.Path.cwd", return_value=tmp_path),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack"),
        ):
            result = runner.invoke(app, ["retrieve", "What changed?"])

        assert result.exit_code == 0
        mock_engine.load.assert_called_once_with(tmp_path.name)
        mock_engine.point.assert_not_called()
        mock_engine.wait_for_query_surface.assert_not_called()
        mock_engine.evidence.assert_called_once()

    def test_spawn_index_daemon_reuses_running_pid(self, tmp_path):
        """A live PID file should prevent duplicate detached daemons."""
        pid_path = tmp_path / ".fitz" / "collections" / "docs" / "index_daemon.pid"
        pid_path.parent.mkdir(parents=True)
        pid_path.write_text("123", encoding="utf-8")

        from fitz_sage.cli.commands import retrieve

        with (
            patch("fitz_sage.cli.commands.retrieve._pid_is_running", return_value=True),
            patch("fitz_sage.cli.commands.retrieve.subprocess.Popen") as popen,
        ):
            spawned = retrieve._spawn_index_daemon("docs", "fitz_krag", tmp_path)

        assert spawned == "running"
        popen.assert_not_called()

    def test_retrieve_spawns_daemon_when_indexing_is_pending(self):
        """CLI query returns evidence, then hands remaining enrichment to a daemon."""
        pack = EvidencePack(
            query="What changed?",
            mode=AnswerMode.SUFFICIENT,
            indexing_status={
                "total": 3,
                "complete": False,
                "fully_enriched": False,
            },
        )

        mock_engine = MagicMock()
        mock_engine.evidence.return_value = pack

        mock_registry = MagicMock()
        mock_registry.list.return_value = ["fitz_krag"]
        mock_caps = MagicMock()
        mock_caps.supports_persistent_ingest = True
        mock_registry.get_capabilities.return_value = mock_caps
        mock_registry.get_list_collections.return_value = ["docs"]

        with (
            patch(
                "fitz_sage.cli.commands.retrieve.get_engine_registry", return_value=mock_registry
            ),
            patch("fitz_sage.cli.commands.retrieve.get_default_engine", return_value="fitz_krag"),
            patch("fitz_sage.cli.commands.retrieve.create_engine", return_value=mock_engine),
            patch("fitz_sage.cli.commands.retrieve.display_evidence_pack"),
            patch("fitz_sage.cli.commands.retrieve._spawn_index_daemon") as spawn,
        ):
            result = runner.invoke(app, ["retrieve", "What changed?", "--collection", "docs"])

        assert result.exit_code == 0
        mock_engine.stop_background_indexing.assert_called_once()
        spawn.assert_called_once()
        assert spawn.call_args.args[0] == "docs"
        assert spawn.call_args.args[1] == "fitz_krag"
