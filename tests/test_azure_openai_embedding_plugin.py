# tests/test_azure_openai_embedding_plugin.py
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest


# Mock response structures
@dataclass
class MockEmbeddingData:
    embedding: list[float]


@dataclass
class MockEmbeddingResponse:
    data: list[MockEmbeddingData]


class TestAzureOpenAIEmbeddingPlugin:
    """Tests for Azure OpenAI embedding plugin."""

    def test_init_with_explicit_params(self):
        """Plugin initializes with explicit parameters."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                client = AzureOpenAIEmbeddingClient(
                    api_key="test-key",
                    endpoint="https://my-resource.openai.azure.com",
                    deployment_name="embedding-deployment",
                )

                assert client.deployment_name == "embedding-deployment"
                mock_azure.assert_called_once()

    def test_init_with_env_vars(self):
        """Plugin initializes with environment variables."""
        env = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env-resource.openai.azure.com",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "env-embed-deployment",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                client = AzureOpenAIEmbeddingClient()

                assert client.deployment_name == "env-embed-deployment"

    def test_init_missing_api_key_raises(self):
        """Plugin raises RuntimeError when no API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI"):
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY"):
                    AzureOpenAIEmbeddingClient(
                        endpoint="https://test.openai.azure.com",
                        deployment_name="test",
                    )

    def test_init_missing_endpoint_raises(self):
        """Plugin raises RuntimeError when no endpoint."""
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "key"}, clear=True):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI"):
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT"):
                    AzureOpenAIEmbeddingClient(deployment_name="test")

    def test_init_missing_deployment_raises(self):
        """Plugin raises RuntimeError when no deployment name."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI"):
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                with pytest.raises(RuntimeError, match="deployment_name"):
                    AzureOpenAIEmbeddingClient()

    def test_init_missing_library_raises(self):
        """Plugin raises RuntimeError when openai not installed."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI", None):
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                with pytest.raises(RuntimeError, match="Install openai"):
                    AzureOpenAIEmbeddingClient(deployment_name="test")

    def test_embed_returns_list_of_floats(self):
        """embed() returns list of floats."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                expected = [0.1, 0.2, 0.3]
                mock_response = MockEmbeddingResponse(data=[MockEmbeddingData(embedding=expected)])
                mock_azure.return_value.embeddings.create.return_value = mock_response

                client = AzureOpenAIEmbeddingClient(deployment_name="test")
                result = client.embed("test text")

                assert result == expected

    def test_embed_uses_deployment_as_model(self):
        """embed() passes deployment_name as model parameter."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                mock_response = MockEmbeddingResponse(data=[MockEmbeddingData(embedding=[0.1])])
                mock_azure.return_value.embeddings.create.return_value = mock_response

                client = AzureOpenAIEmbeddingClient(deployment_name="my-embedding")
                client.embed("test")

                call_kwargs = mock_azure.return_value.embeddings.create.call_args[1]
                assert call_kwargs["model"] == "my-embedding"

    def test_embed_includes_dimensions_when_set(self):
        """embed() includes dimensions in API call when specified."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                mock_response = MockEmbeddingResponse(data=[MockEmbeddingData(embedding=[0.1])])
                mock_azure.return_value.embeddings.create.return_value = mock_response

                client = AzureOpenAIEmbeddingClient(
                    deployment_name="test",
                    dimensions=512,
                )
                client.embed("test")

                call_kwargs = mock_azure.return_value.embeddings.create.call_args[1]
                assert call_kwargs["dimensions"] == 512

    def test_embed_raises_embedding_error_on_failure(self):
        """embed() raises EmbeddingError on API failure."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from fitz.engines.classic_rag.errors.llm import EmbeddingError
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                mock_azure.return_value.embeddings.create.side_effect = Exception("API Error")

                client = AzureOpenAIEmbeddingClient(deployment_name="test")

                with pytest.raises(EmbeddingError, match="Failed to embed"):
                    client.embed("test text")

    def test_plugin_attributes(self):
        """Plugin has correct plugin_name and plugin_type."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("fitz.core.llm.embedding.plugins.azure_openai.AzureOpenAI"):
                from fitz.llm.embedding.plugins.azure_openai import (
                    AzureOpenAIEmbeddingClient,
                )

                client = AzureOpenAIEmbeddingClient(deployment_name="test")

                assert client.plugin_name == "azure_openai"
                assert client.plugin_type == "embedding"
