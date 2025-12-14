# tests/test_openai_embedding_plugin.py
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# Mock response structures
@dataclass
class MockEmbeddingData:
    embedding: list[float]


@dataclass
class MockEmbeddingResponse:
    data: list[MockEmbeddingData]


class TestOpenAIEmbeddingPlugin:
    """Tests for OpenAI embedding plugin."""

    def test_init_with_explicit_api_key(self):
        """Plugin initializes with explicit API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                client = OpenAIEmbeddingClient(api_key="test-key")

                assert client.model == "text-embedding-3-small"
                mock_openai.assert_called_once()

    def test_init_with_env_var(self):
        """Plugin initializes with OPENAI_API_KEY env var."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                client = OpenAIEmbeddingClient()

                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["api_key"] == "env-key"

    def test_init_missing_api_key_raises(self):
        """Plugin raises RuntimeError when no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.embedding.plugins.openai.OpenAI"):
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                    OpenAIEmbeddingClient()

    def test_init_missing_library_raises(self):
        """Plugin raises RuntimeError when openai not installed."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI", None):
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                with pytest.raises(RuntimeError, match="Install openai"):
                    OpenAIEmbeddingClient()

    def test_init_with_custom_model(self):
        """Plugin accepts custom model."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI"):
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                client = OpenAIEmbeddingClient(model="text-embedding-3-large")

                assert client.model == "text-embedding-3-large"

    def test_init_with_dimensions(self):
        """Plugin accepts custom dimensions."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI"):
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                client = OpenAIEmbeddingClient(dimensions=512)

                assert client.dimensions == 512

    def test_init_with_base_url(self):
        """Plugin accepts custom base_url."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                OpenAIEmbeddingClient(base_url="https://custom.api.com/v1")

                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["base_url"] == "https://custom.api.com/v1"

    def test_embed_returns_list_of_floats(self):
        """embed() returns list of floats."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
                mock_response = MockEmbeddingResponse(
                    data=[MockEmbeddingData(embedding=expected_embedding)]
                )
                mock_openai.return_value.embeddings.create.return_value = mock_response

                client = OpenAIEmbeddingClient()
                result = client.embed("Hello world")

                assert result == expected_embedding
                assert isinstance(result, list)
                assert all(isinstance(x, float) for x in result)

    def test_embed_passes_model_to_api(self):
        """embed() passes correct model to API."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                mock_response = MockEmbeddingResponse(
                    data=[MockEmbeddingData(embedding=[0.1])]
                )
                mock_openai.return_value.embeddings.create.return_value = mock_response

                client = OpenAIEmbeddingClient(model="text-embedding-3-large")
                client.embed("test")

                call_kwargs = mock_openai.return_value.embeddings.create.call_args[1]
                assert call_kwargs["model"] == "text-embedding-3-large"

    def test_embed_includes_dimensions_when_set(self):
        """embed() includes dimensions in API call when specified."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                mock_response = MockEmbeddingResponse(
                    data=[MockEmbeddingData(embedding=[0.1])]
                )
                mock_openai.return_value.embeddings.create.return_value = mock_response

                client = OpenAIEmbeddingClient(dimensions=256)
                client.embed("test")

                call_kwargs = mock_openai.return_value.embeddings.create.call_args[1]
                assert call_kwargs["dimensions"] == 256

    def test_embed_raises_embedding_error_on_failure(self):
        """embed() raises EmbeddingError on API failure."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient
                from core.exceptions.llm import EmbeddingError

                mock_openai.return_value.embeddings.create.side_effect = Exception("API Error")

                client = OpenAIEmbeddingClient()

                with pytest.raises(EmbeddingError, match="Failed to embed"):
                    client.embed("test text")

    def test_init_raises_embedding_error_on_client_failure(self):
        """Plugin raises EmbeddingError when client init fails."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI") as mock_openai:
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient
                from core.exceptions.llm import EmbeddingError

                mock_openai.side_effect = Exception("Connection failed")

                with pytest.raises(EmbeddingError, match="Failed to initialize"):
                    OpenAIEmbeddingClient()

    def test_plugin_attributes(self):
        """Plugin has correct plugin_name and plugin_type."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.embedding.plugins.openai.OpenAI"):
                from core.llm.embedding.plugins.openai import OpenAIEmbeddingClient

                client = OpenAIEmbeddingClient()

                assert client.plugin_name == "openai"
                assert client.plugin_type == "embedding"
