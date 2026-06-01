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

    def test_retrieve_json_outputs_evidence_pack_only(self):
        """JSON mode should emit a loadable EvidencePack without progress text."""
        pack = EvidencePack(
            query="Which test failed?",
            mode=AnswerMode.TRUSTWORTHY,
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
        assert data["mode"] == "trustworthy"
        mock_engine.load.assert_called_once_with("test")
        mock_engine.evidence.assert_called_once()
