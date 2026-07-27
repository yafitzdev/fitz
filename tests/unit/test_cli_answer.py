# tests/unit/test_cli_answer.py
"""
Unit tests for `fitz answer --endpoint / --synthesizer / --model / --api-key-env` flags.

These flags let users point the CLI at any OpenAI-compatible HTTP
server without editing engine YAML — the canonical UX for the
single-protocol architecture.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import typer

from fitz_sage.cli.commands.answer import _apply_chat_overrides


@pytest.fixture(autouse=True)
def _skip_firstrun():
    with patch("fitz_sage.config.firstrun.needs_firstrun", return_value=False):
        yield


class TestApplyChatOverrides:
    """The override helper builds a config-with-overrides per the flags."""

    def test_no_flags_returns_none(self) -> None:
        """No flags = no override = engine loads its own config."""
        result = _apply_chat_overrides("fitz_krag", None, None, None)
        assert result is None

    def test_unknown_engine_warns_and_returns_none(self) -> None:
        """Unsupported engines fall through with a warning."""
        with patch("fitz_sage.cli.commands.answer.ui") as mock_ui:
            result = _apply_chat_overrides(
                "some_other_engine",
                "http://localhost:8080/v1",
                "qwen2.5-7b",
                None,
            )
        assert result is None
        mock_ui.warning.assert_called_once()

    def test_endpoint_with_model(self) -> None:
        """--endpoint + --model rewrite chat_* tiers + chat_base_url."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.chat_smart = "endpoint/qwen2.5-7b-instruct"
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "http://localhost:8080/v1",
                "qwen2.5-7b",
                None,
            )

        mock_config.model_copy.assert_called_once_with(
            update={
                "chat_base_url": "http://localhost:8080/v1",
                "chat_fast": "endpoint/qwen2.5-7b",
                "chat_balanced": "endpoint/qwen2.5-7b",
                "chat_smart": "endpoint/qwen2.5-7b",
                "synthesizer": "endpoint/qwen2.5-7b",
            }
        )

    def test_synthesizer_only_sets_role_provider(self) -> None:
        """--synthesizer overrides only the synthesis provider."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                None,
                None,
                None,
                synthesizer="openai/gpt-4o",
            )

        mock_config.model_copy.assert_called_once_with(update={"synthesizer": "openai/gpt-4o"})

    def test_synthesizer_endpoint_pairs_with_endpoint(self) -> None:
        """--synthesizer endpoint/<model> can take the URL from --endpoint."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "http://localhost:8080/v1",
                None,
                None,
                synthesizer="endpoint/qwen2.5-7b",
            )

        mock_config.model_copy.assert_called_once_with(
            update={
                "chat_base_url": "http://localhost:8080/v1",
                "synthesizer": "endpoint/qwen2.5-7b",
            }
        )

    def test_synthesizer_endpoint_requires_endpoint_or_configured_url(self) -> None:
        """Direct endpoint specs need a URL from the CLI or config."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.chat_base_url = None
        mock_registry.load_config.return_value = mock_config

        with (
            patch(
                "fitz_sage.runtime.registry.get_engine_registry",
                return_value=mock_registry,
            ),
            patch("fitz_sage.cli.commands.answer.ui"),
            pytest.raises(typer.Exit),
        ):
            _apply_chat_overrides(
                "fitz_krag",
                None,
                None,
                None,
                synthesizer="endpoint/qwen2.5-7b",
            )

    def test_synthesizer_and_model_conflict(self) -> None:
        """--synthesizer and --model both set the synthesis model."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with (
            patch(
                "fitz_sage.runtime.registry.get_engine_registry",
                return_value=mock_registry,
            ),
            patch("fitz_sage.cli.commands.answer.ui"),
            pytest.raises(typer.Exit),
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "http://localhost:8080/v1",
                "qwen2.5-7b",
                None,
                synthesizer="endpoint/qwen2.5-14b",
            )

    def test_endpoint_without_model_reuses_existing_synthesizer_model(self) -> None:
        """--endpoint alone keeps the existing model name, swaps the URL."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.synthesizer = "endpoint/qwen2.5-14b-instruct"
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "http://localhost:8080/v1",
                None,
                None,
            )

        update = mock_config.model_copy.call_args.kwargs["update"]
        assert update["chat_base_url"] == "http://localhost:8080/v1"
        assert update["chat_smart"] == "endpoint/qwen2.5-14b-instruct"
        assert update["synthesizer"] == "endpoint/qwen2.5-14b-instruct"

    def test_endpoint_without_model_existing_no_slash(self) -> None:
        """If existing synthesizer has no slash, use it as the bare model name."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.synthesizer = "qwen2.5-14b-instruct"
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "http://localhost:8080/v1",
                None,
                None,
            )

        update = mock_config.model_copy.call_args.kwargs["update"]
        assert update["chat_smart"] == "endpoint/qwen2.5-14b-instruct"
        assert update["synthesizer"] == "endpoint/qwen2.5-14b-instruct"

    def test_endpoint_without_model_and_no_configured_model_exits(self) -> None:
        """--endpoint alone needs a model when config has no synthesizer."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.synthesizer = None
        mock_registry.load_config.return_value = mock_config

        with (
            patch(
                "fitz_sage.runtime.registry.get_engine_registry",
                return_value=mock_registry,
            ),
            patch("fitz_sage.cli.commands.answer.ui"),
            pytest.raises(typer.Exit),
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "http://localhost:8080/v1",
                None,
                None,
            )

    def test_api_key_env_only(self) -> None:
        """--api-key-env without --endpoint still applies."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.chat_smart = "endpoint/qwen2.5-7b"
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                None,
                None,
                "TOGETHER_API_KEY",
            )

        update = mock_config.model_copy.call_args.kwargs["update"]
        assert update == {"chat_api_key_env": "TOGETHER_API_KEY"}

    def test_all_three_flags(self) -> None:
        """All three flags compose cleanly (cloud-style endpoint with auth)."""
        mock_registry = MagicMock()
        mock_config = MagicMock()
        mock_config.chat_smart = "endpoint/qwen2.5-7b"
        mock_config.model_copy.return_value = MagicMock()
        mock_registry.load_config.return_value = mock_config

        with patch(
            "fitz_sage.runtime.registry.get_engine_registry",
            return_value=mock_registry,
        ):
            _apply_chat_overrides(
                "fitz_krag",
                "https://api.together.xyz/v1",
                "meta-llama-3.1-70b",
                "TOGETHER_API_KEY",
            )

        update = mock_config.model_copy.call_args.kwargs["update"]
        assert update["chat_base_url"] == "https://api.together.xyz/v1"
        assert update["chat_smart"] == "endpoint/meta-llama-3.1-70b"
        assert update["synthesizer"] == "endpoint/meta-llama-3.1-70b"
        assert update["chat_api_key_env"] == "TOGETHER_API_KEY"


class TestAnswerCommandFlagPlumbing:
    """End-to-end flag plumbing through the CLI runner."""

    def test_retrieve_help_has_no_synthesis_flags(self) -> None:
        import re

        from typer.testing import CliRunner

        from fitz_sage.cli.cli import app

        result = CliRunner().invoke(app, ["retrieve", "--help"])
        assert result.exit_code == 0
        # CI runners emit ANSI escape codes inside the rendered help (rich
        # interleaves color codes between the two dashes of long flags),
        # which breaks substring matching. Strip them before asserting.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--source" in plain
        assert "--collection" in plain
        assert "--endpoint" not in plain
        assert "--synthesizer" not in plain
        assert "--model" not in plain
        assert "--api-key-env" not in plain

    def test_answer_help_exposes_synthesizer_flag(self) -> None:
        import re

        from typer.testing import CliRunner

        from fitz_sage.cli.cli import app

        result = CliRunner().invoke(app, ["answer", "--help"])
        assert result.exit_code == 0
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--endpoint" in plain
        assert "--synthesizer" in plain
        assert "--model" in plain
        assert "--api-key-env" in plain
