# tests/unit/llm/test_endpoint_provider.py
"""
Unit tests for the OpenAI-compatible 'endpoint' provider.

The endpoint provider is a thin wrapper around OpenAIChat/OpenAIEmbedding/
OpenAIVision that points at any OpenAI-compatible HTTP server (llama-server,
vLLM, LM Studio, OpenRouter, Together, etc.) without requiring the user
to know which "real" provider is behind the URL.

Auth defaults to NoAuth (for local servers); opt-in to API key via
config['auth']['api_key_env'].
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

from fitz_sage.llm.auth import ApiKeyAuth, NoAuth
from fitz_sage.llm.config import (
    create_chat_provider,
    create_embedding_provider,
    create_vision_provider,
    resolve_auth,
)

HAS_OPENAI = importlib.util.find_spec("openai") is not None


class TestEndpointAuthResolution:
    """Test auth resolution for the endpoint provider."""

    def test_no_auth_by_default(self) -> None:
        """Endpoint provider defaults to NoAuth (for local servers)."""
        auth = resolve_auth("endpoint")
        assert isinstance(auth, NoAuth)

    def test_no_auth_with_empty_config(self) -> None:
        """Empty config also resolves to NoAuth."""
        auth = resolve_auth("endpoint", {})
        assert isinstance(auth, NoAuth)

    def test_no_auth_returns_empty_headers(self) -> None:
        """NoAuth produces no auth headers."""
        auth = resolve_auth("endpoint")
        assert auth is not None
        assert auth.get_headers() == {}
        assert auth.get_request_kwargs() == {}

    def test_api_key_opt_in(self) -> None:
        """Setting auth.api_key_env opts into ApiKeyAuth."""
        config = {"auth": {"api_key_env": "MY_ENDPOINT_KEY"}}
        auth = resolve_auth("endpoint", config)
        assert isinstance(auth, ApiKeyAuth)
        assert auth.env_var == "MY_ENDPOINT_KEY"
        assert auth.header_format == "bearer"

    def test_api_key_with_custom_header_format(self) -> None:
        """Custom header_format is respected."""
        config = {
            "auth": {
                "api_key_env": "MY_KEY",
                "header_format": "x-api-key",
            }
        }
        auth = resolve_auth("endpoint", config)
        assert isinstance(auth, ApiKeyAuth)
        assert auth.header_format == "x-api-key"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestEndpointChatProvider:
    """Test create_chat_provider for the endpoint provider."""

    def test_basic_chat_creation(self) -> None:
        """Endpoint chat with base_url + model creates an OpenAIChat."""
        with patch("openai.OpenAI") as mock_openai:
            provider = create_chat_provider(
                "endpoint/qwen2.5-7b",
                config={"base_url": "http://localhost:8080/v1"},
            )
            assert provider._model == "qwen2.5-7b"
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8080/v1"

    def test_missing_base_url_raises(self) -> None:
        """No base_url is a configuration error."""
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="endpoint provider requires 'base_url'"):
                create_chat_provider("endpoint/qwen2.5-7b")

    def test_missing_model_raises(self) -> None:
        """No model in spec is a configuration error."""
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="endpoint provider requires a model"):
                create_chat_provider(
                    "endpoint",
                    config={"base_url": "http://localhost:8080/v1"},
                )

    def test_uses_no_auth_by_default(self) -> None:
        """Without auth config, no API key is required at construction time."""
        # No env vars set, no api_key_env in config — should still work
        with patch("openai.OpenAI"):
            provider = create_chat_provider(
                "endpoint/local-model",
                config={"base_url": "http://localhost:8080/v1"},
            )
            assert provider._model == "local-model"

    def test_api_key_auth_is_used_when_configured(self) -> None:
        """When auth.api_key_env is set, ApiKeyAuth is wired in."""
        config = {
            "base_url": "https://api.together.xyz/v1",
            "auth": {"api_key_env": "TOGETHER_API_KEY"},
        }
        with patch.dict("os.environ", {"TOGETHER_API_KEY": "sk-test"}):
            with patch("openai.OpenAI"):
                provider = create_chat_provider(
                    "endpoint/meta-llama-3.1-70b",
                    config=config,
                )
                assert provider._model == "meta-llama-3.1-70b"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestEndpointEmbeddingProvider:
    """Test create_embedding_provider for the endpoint provider."""

    def test_basic_embedding_creation(self) -> None:
        """Endpoint embedding with base_url + model creates an OpenAIEmbedding."""
        with patch("openai.OpenAI") as mock_openai:
            provider = create_embedding_provider(
                "endpoint/nomic-embed-text",
                config={"base_url": "http://localhost:8081/v1"},
            )
            assert provider._model == "nomic-embed-text"
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8081/v1"

    def test_missing_base_url_raises(self) -> None:
        """No base_url is a configuration error."""
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="endpoint provider requires 'base_url'"):
                create_embedding_provider("endpoint/nomic-embed-text")

    def test_missing_model_raises(self) -> None:
        """No model in spec is a configuration error."""
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="endpoint provider requires a model"):
                create_embedding_provider(
                    "endpoint",
                    config={"base_url": "http://localhost:8081/v1"},
                )

    def test_dimensions_passthrough(self) -> None:
        """Custom dimensions are forwarded to the underlying provider."""
        with patch("openai.OpenAI"):
            provider = create_embedding_provider(
                "endpoint/embed-model",
                config={
                    "base_url": "http://localhost:8081/v1",
                    "dimensions": 512,
                },
            )
            assert provider._dimensions == 512


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestEndpointVisionProvider:
    """Test create_vision_provider for the endpoint provider."""

    def test_basic_vision_creation(self) -> None:
        """Endpoint vision with base_url + model creates an OpenAIVision."""
        with patch("openai.OpenAI") as mock_openai:
            provider = create_vision_provider(
                "endpoint/qwen2-vl-7b",
                config={"base_url": "http://localhost:8080/v1"},
            )
            assert provider is not None
            assert provider._model == "qwen2-vl-7b"
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8080/v1"

    def test_missing_base_url_raises(self) -> None:
        """No base_url is a configuration error."""
        with patch("openai.OpenAI"):
            with pytest.raises(ValueError, match="endpoint provider requires 'base_url'"):
                create_vision_provider("endpoint/qwen2-vl-7b")
