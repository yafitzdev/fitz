# tests/unit/llm/test_config.py
"""
Unit tests for LLM config parser.

Provider-specific tests require the SDK to be installed.
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

from fitz_sage.llm.auth import ApiKeyAuth, M2MAuth, NoAuth
from fitz_sage.llm.config import (
    create_chat_provider,
    create_rerank_provider,
    create_vision_provider,
    parse_provider_string,
    resolve_auth,
)

# Check for optional SDKs
HAS_OPENAI = importlib.util.find_spec("openai") is not None


class TestParseProviderString:
    """Test provider string parsing."""

    def test_provider_only(self) -> None:
        """Parse provider without model."""
        provider, model = parse_provider_string("cohere")
        assert provider == "cohere"
        assert model is None

    def test_provider_with_model(self) -> None:
        """Parse provider with model."""
        provider, model = parse_provider_string("cohere/command-a-03-2025")
        assert provider == "cohere"
        assert model == "command-a-03-2025"

    def test_openai_with_model(self) -> None:
        """Parse OpenAI provider with model."""
        provider, model = parse_provider_string("openai/gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_strips_whitespace(self) -> None:
        """Whitespace is stripped."""
        provider, model = parse_provider_string("  cohere / command-a-03-2025  ")
        assert provider == "cohere"
        assert model == "command-a-03-2025"

    def test_model_with_slashes(self) -> None:
        """Model names can contain slashes (only first split)."""
        provider, model = parse_provider_string("huggingface/meta-llama/Llama-2-70b")
        assert provider == "huggingface"
        assert model == "meta-llama/Llama-2-70b"


class TestResolveAuth:
    """Test auth resolution."""

    def test_openai_api_key(self) -> None:
        """OpenAI preset uses OPENAI_API_KEY with bearer format."""
        auth = resolve_auth("openai")
        assert isinstance(auth, ApiKeyAuth)
        assert auth.env_var == "OPENAI_API_KEY"
        assert auth.header_format == "bearer"

    def test_azure_openai_api_key(self) -> None:
        """Azure OpenAI preset uses AZURE_OPENAI_API_KEY with bearer format."""
        auth = resolve_auth("azure_openai")
        assert isinstance(auth, ApiKeyAuth)
        assert auth.env_var == "AZURE_OPENAI_API_KEY"
        assert auth.header_format == "bearer"

    def test_endpoint_no_auth_default(self) -> None:
        """Endpoint defaults to NoAuth (for local servers)."""
        auth = resolve_auth("endpoint")
        assert isinstance(auth, NoAuth)

    def test_m2m_auth(self, temp_certificate) -> None:
        """M2M auth is created from config."""
        cert_path, _ = temp_certificate
        config = {
            "auth": {
                "type": "m2m",
                "token_url": "https://auth.example.com/token",
                "client_id": "my-client",
                "client_secret": "my-secret",
                "scope": "read write",
            },
            "cert_path": cert_path,
        }
        auth = resolve_auth("openai", config)
        assert isinstance(auth, M2MAuth)
        assert auth.token_url == "https://auth.example.com/token"
        assert auth.client_id == "my-client"
        assert auth.client_secret == "my-secret"
        assert auth.cert_path == cert_path
        assert auth.scope == "read write"

    def test_m2m_auth_accepts_cert_path_inside_auth(self, temp_certificate) -> None:
        """KRAG YAML can keep enterprise certificate settings in auth."""
        cert_path, _ = temp_certificate
        config = {
            "auth": {
                "type": "m2m",
                "token_url": "https://auth.example.com/token",
                "client_id": "my-client",
                "client_secret": "my-secret",
                "cert_path": cert_path,
            },
        }

        auth = resolve_auth("openai", config)

        assert isinstance(auth, M2MAuth)
        assert auth.cert_path == cert_path

    def test_m2m_auth_minimal(self) -> None:
        """M2M auth works with minimal config."""
        config = {
            "auth": {
                "type": "m2m",
                "token_url": "https://auth.example.com/token",
                "client_id": "my-client",
                "client_secret": "my-secret",
            }
        }
        auth = resolve_auth("openai", config)
        assert isinstance(auth, M2MAuth)
        assert auth.cert_path is None
        assert auth.scope is None


class TestUnknownProvider:
    """Test unknown provider handling."""

    def test_unknown_chat_provider_raises(self) -> None:
        """Unknown provider raises ValueError with supported list."""
        with pytest.raises(ValueError, match="Unknown chat provider: unknown"):
            create_chat_provider("unknown")

    def test_onnx_chat_provider_builds(self) -> None:
        """`onnx` dispatches the managed Qwen ONNX chat provider."""
        from fitz_sage.llm.providers.onnx_chat import DEFAULT_QWEN_MODEL_ID, OnnxChat

        chat = create_chat_provider("onnx/qwen3-0.6b")
        assert isinstance(chat, OnnxChat)
        assert chat._snapshot.spec.repo_id == DEFAULT_QWEN_MODEL_ID

    def test_unknown_rerank_provider_raises(self) -> None:
        """Unknown rerank provider raises with the supported list."""
        with pytest.raises(ValueError, match="Unknown rerank provider"):
            create_rerank_provider("anything")

    def test_onnx_rerank_provider_builds(self) -> None:
        """`onnx` dispatches the OnnxReranker with the default model."""
        from fitz_sage.llm.providers.onnx_reranker import DEFAULT_MODEL_ID, OnnxReranker

        reranker = create_rerank_provider("onnx")
        assert isinstance(reranker, OnnxReranker)
        assert reranker._model_id == DEFAULT_MODEL_ID

    def test_unknown_vision_provider_raises(self) -> None:
        """Unknown vision provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vision provider: weird"):
            create_vision_provider("weird")


class TestNoneHandling:
    """Test None handling."""

    def test_rerank_none_returns_none(self) -> None:
        """None spec returns None."""
        provider = create_rerank_provider(None)
        assert provider is None

    def test_vision_none_returns_none(self) -> None:
        """None spec returns None."""
        provider = create_vision_provider(None)
        assert provider is None


# Provider-specific tests - only run if SDK is installed


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestOpenAIPreset:
    """Test the 'openai' provider name as a preset over OpenAICompat."""

    def test_chat_provider_uses_default_base_url(self) -> None:
        """`openai` preset routes to OpenAICompatChat with the public OpenAI URL."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                provider = create_chat_provider("openai")
                assert provider._model == "gpt-4o"
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["base_url"] == "https://api.openai.com/v1"

    def test_chat_with_base_url_override(self) -> None:
        """User-supplied base_url overrides the OpenAI default."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                create_chat_provider("openai", config={"base_url": "https://api.proxy.com/v1"})
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["base_url"] == "https://api.proxy.com/v1"

    def test_vision_provider(self) -> None:
        """`openai/<model>` produces an OpenAICompatVision."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                provider = create_vision_provider("openai/gpt-4o")
                assert provider is not None
                assert provider._model == "gpt-4o"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestAzureOpenAIPreset:
    """Test the 'azure_openai' provider name as a preset."""

    def test_chat_requires_base_url(self) -> None:
        """Azure base URLs are tenant-specific; missing one is an error."""
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                with pytest.raises(ValueError, match="azure_openai requires 'base_url'"):
                    create_chat_provider("azure_openai/my-deployment")

    def test_chat_with_base_url(self) -> None:
        """Azure chat with explicit base URL works."""
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI") as mock_openai:
                provider = create_chat_provider(
                    "azure_openai/my-deployment",
                    config={
                        "base_url": "https://my-tenant.openai.azure.com/openai/deployments/my-deployment"
                    },
                )
                assert provider._model == "my-deployment"
                assert (
                    mock_openai.call_args[1]["base_url"]
                    == "https://my-tenant.openai.azure.com/openai/deployments/my-deployment"
                )
