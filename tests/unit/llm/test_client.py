# tests/unit/llm/test_client.py
"""
Unit tests for the public LLM client API.

There is exactly one chat-protocol implementation in fitz-sage —
``OpenAICompatChat`` — reachable through three provider names:
``endpoint`` (canonical, BYO URL), ``openai`` (preset for the public
OpenAI API), and ``azure_openai`` (preset for tenant-specific Azure).
"""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

from fitz_sage.llm.client import get_chat, get_reranker, get_vision
from fitz_sage.llm.providers import ChatProvider, VisionProvider

HAS_OPENAI = importlib.util.find_spec("openai") is not None


class TestGetReranker:
    """get_reranker honours None and dispatches the 'onnx' cross-encoder."""

    def test_none_returns_none(self) -> None:
        assert get_reranker(None) is None

    def test_unknown_spec_raises(self) -> None:
        """Unknown rerank spec raises with the supported list."""
        with pytest.raises(ValueError, match="Unknown rerank provider"):
            get_reranker("anything")

    def test_onnx_rerank_provider_builds(self) -> None:
        """The 'onnx' rerank spec instantiates an OnnxReranker."""
        from fitz_sage.llm.providers.onnx_reranker import DEFAULT_MODEL_ID, OnnxReranker

        reranker = get_reranker("onnx")
        assert isinstance(reranker, OnnxReranker)
        assert reranker._model_id == DEFAULT_MODEL_ID

    def test_onnx_rerank_custom_model(self) -> None:
        """`onnx/<hf-model-id>` passes the model id through."""
        from fitz_sage.llm.providers.onnx_reranker import OnnxReranker

        reranker = get_reranker("onnx/BAAI/bge-reranker-base")
        assert isinstance(reranker, OnnxReranker)
        assert reranker._model_id == "BAAI/bge-reranker-base"


class TestGetVision:
    """get_vision honours None."""

    def test_none_returns_none(self) -> None:
        assert get_vision(None) is None


class TestUnknownProvider:
    """Unknown provider names raise actionable errors."""

    def test_unknown_chat_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chat provider"):
            get_chat("unknown_provider")

    def test_onnx_chat_provider_builds(self) -> None:
        """The managed ONNX Qwen provider is available through get_chat."""
        from fitz_sage.llm.providers.onnx_chat import DEFAULT_QWEN_MODEL_ID, OnnxChat

        chat = get_chat("onnx/qwen3.5-0.8b")
        assert isinstance(chat, OnnxChat)
        assert chat._model_id == DEFAULT_QWEN_MODEL_ID

    def test_unknown_vision_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown vision provider"):
            get_vision("unknown_provider")


class TestRemovedProviderErrors:
    """Removed provider names raise actionable errors."""

    @pytest.mark.parametrize("removed", ["ollama", "cohere", "anthropic"])
    def test_get_chat_raises(self, removed: str) -> None:
        with pytest.raises(ValueError, match=f"'{removed}' provider has been removed"):
            get_chat(removed)

    @pytest.mark.parametrize("removed", ["ollama", "cohere", "anthropic"])
    def test_get_reranker_raises(self, removed: str) -> None:
        with pytest.raises(ValueError, match=f"'{removed}' provider has been removed"):
            get_reranker(removed)

    @pytest.mark.parametrize("removed", ["ollama", "cohere", "anthropic"])
    def test_get_vision_raises(self, removed: str) -> None:
        with pytest.raises(ValueError, match=f"'{removed}' provider has been removed"):
            get_vision(removed)


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestOpenAIPreset:
    """`openai` and `azure_openai` are presets over OpenAICompat."""

    def test_openai_chat(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                chat = get_chat("openai")
                assert isinstance(chat, ChatProvider)

    def test_openai_vision(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.OpenAI"):
                vision = get_vision("openai/gpt-4o")
                assert vision is not None
                assert isinstance(vision, VisionProvider)


@pytest.mark.skipif(not HAS_OPENAI, reason="openai SDK not installed")
class TestEndpointProvider:
    """The canonical 'endpoint' provider — bring your own URL."""

    def test_endpoint_chat_local(self) -> None:
        """Local endpoint server with no auth."""
        with patch("openai.OpenAI"):
            chat = get_chat(
                "endpoint/qwen2.5-7b",
                config={"base_url": "http://localhost:8080/v1"},
            )
            assert isinstance(chat, ChatProvider)

    def test_endpoint_vision_local(self) -> None:
        """Local vision-capable server with no auth."""
        with patch("openai.OpenAI"):
            vision = get_vision(
                "endpoint/qwen2-vl-7b",
                config={"base_url": "http://localhost:8080/v1"},
            )
            assert vision is not None
            assert isinstance(vision, VisionProvider)

    def test_endpoint_chat_cloud_with_key(self) -> None:
        """Cloud OpenAI-compatible with an API key (e.g. Together)."""
        with patch.dict("os.environ", {"TOGETHER_API_KEY": "sk-test"}):
            with patch("openai.OpenAI"):
                chat = get_chat(
                    "endpoint/meta-llama-3.1-70b",
                    config={
                        "base_url": "https://api.together.xyz/v1",
                        "auth": {"api_key_env": "TOGETHER_API_KEY"},
                    },
                )
                assert isinstance(chat, ChatProvider)
