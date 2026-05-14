# tests/unit/test_firstrun.py
"""
Unit tests for first-run auto-configuration.

The new behavior probes common ports for an OpenAI-compatible server
(/v1/models), then falls back to OpenAI cloud if OPENAI_API_KEY is
set. There is no Ollama-specific path — Ollama users speak /v1/
just like everyone else. fitz-sage uses no embeddings, so the
generated config only writes a chat model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fitz_sage.core.firstrun import (
    DetectedEndpoint,
    EndpointModel,
    detect_endpoint,
    run_firstrun_setup,
)


class TestDetectEndpoint:
    """detect_endpoint probes ports and reads /v1/models."""

    def test_no_server_returns_none(self) -> None:
        """If no port responds, detection returns None."""
        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert detect_endpoint() is None

    def test_finds_chat_only_server(self) -> None:
        """A server with one chat model populates chat_models."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "qwen2.5-7b-instruct"}]}

        # First probe responds; rest do not.
        responses = [mock_response] + [Exception("nope")] * 20

        def fake_get(*args, **kwargs):
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        with patch("httpx.get", side_effect=fake_get):
            endpoint = detect_endpoint()

        assert endpoint is not None
        assert endpoint.base_url.endswith("/v1")
        assert len(endpoint.chat_models) == 1
        assert endpoint.chat_models[0].id == "qwen2.5-7b-instruct"

    def test_listing_with_multiple_models(self) -> None:
        """All listed models become chat candidates (no embedding split)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "qwen2.5-7b-instruct"},
                {"id": "nomic-embed-text-v1.5"},
            ]
        }
        with patch("httpx.get", return_value=mock_response):
            endpoint = detect_endpoint()

        assert endpoint is not None
        ids = [m.id for m in endpoint.chat_models]
        assert "qwen2.5-7b-instruct" in ids
        assert "nomic-embed-text-v1.5" in ids

    def test_handles_models_field_alternative(self) -> None:
        """Servers that put models under 'models' (not 'data') still work."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3"}, {"name": "bge-base"}]}
        with patch("httpx.get", return_value=mock_response):
            endpoint = detect_endpoint()

        assert endpoint is not None
        ids = [m.id for m in endpoint.chat_models]
        assert "llama3" in ids
        assert "bge-base" in ids


class TestRunFirstrunSetup:
    """End-to-end first-run flow picks the right configuration source."""

    def test_local_endpoint_wins(self, tmp_path) -> None:
        """When a local OpenAI-compatible server is reachable, use it."""
        endpoint = DetectedEndpoint(
            base_url="http://localhost:8080/v1",
            chat_models=[EndpointModel(id="qwen2.5-7b-instruct")],
        )
        with (
            patch("fitz_sage.core.firstrun.detect_endpoint", return_value=endpoint),
            patch(
                "fitz_sage.core.firstrun.FitzPaths.config",
                return_value=tmp_path / "config.yaml",
            ),
        ):
            ok = run_firstrun_setup()

        assert ok is True
        config = (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert "endpoint/qwen2.5-7b-instruct" in config
        assert "chat_base_url: http://localhost:8080/v1" in config

    def test_openai_key_fallback(self, tmp_path, monkeypatch) -> None:
        """No local endpoint + OPENAI_API_KEY -> openai preset config."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with (
            patch("fitz_sage.core.firstrun.detect_endpoint", return_value=None),
            patch(
                "fitz_sage.core.firstrun.FitzPaths.config",
                return_value=tmp_path / "config.yaml",
            ),
        ):
            ok = run_firstrun_setup()

        assert ok is True
        config = (tmp_path / "config.yaml").read_text(encoding="utf-8")
        assert "openai/gpt-4o" in config

    def test_no_provider_aborts(self, tmp_path, monkeypatch) -> None:
        """No endpoint, no key -> setup fails with instructions."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with (
            patch("fitz_sage.core.firstrun.detect_endpoint", return_value=None),
            patch(
                "fitz_sage.core.firstrun.FitzPaths.config",
                return_value=tmp_path / "config.yaml",
            ),
        ):
            ok = run_firstrun_setup()

        assert ok is False
        assert not (tmp_path / "config.yaml").exists()

    def test_endpoint_with_no_chat_models_aborts(self, tmp_path) -> None:
        """A reachable server with no models is a configuration error."""
        endpoint = DetectedEndpoint(
            base_url="http://localhost:8080/v1",
            chat_models=[],
        )
        with (
            patch("fitz_sage.core.firstrun.detect_endpoint", return_value=endpoint),
            patch(
                "fitz_sage.core.firstrun.FitzPaths.config",
                return_value=tmp_path / "config.yaml",
            ),
        ):
            ok = run_firstrun_setup()

        assert ok is False
        assert not (tmp_path / "config.yaml").exists()
