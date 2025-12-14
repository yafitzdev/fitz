# tests/test_azure_openai_chat_plugin.py
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# Mock response structures
@dataclass
class MockMessage:
    content: str


@dataclass
class MockChoice:
    message: MockMessage


@dataclass
class MockChatCompletion:
    choices: list


class TestAzureOpenAIChatPlugin:
    """Tests for Azure OpenAI chat plugin."""

    def test_init_with_explicit_params(self):
        """Plugin initializes with explicit parameters."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                client = AzureOpenAIChatClient(
                    api_key="test-key",
                    endpoint="https://my-resource.openai.azure.com",
                    deployment_name="gpt4-deployment",
                )

                assert client.deployment_name == "gpt4-deployment"
                assert client.temperature == 0.2
                mock_azure.assert_called_once()

    def test_init_with_env_vars(self):
        """Plugin initializes with environment variables."""
        env = {
            "AZURE_OPENAI_API_KEY": "env-key",
            "AZURE_OPENAI_ENDPOINT": "https://env-resource.openai.azure.com",
            "AZURE_OPENAI_DEPLOYMENT": "env-deployment",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                client = AzureOpenAIChatClient()

                assert client.deployment_name == "env-deployment"
                mock_azure.assert_called_once()
                call_kwargs = mock_azure.call_args[1]
                assert call_kwargs["api_key"] == "env-key"
                assert call_kwargs["azure_endpoint"] == "https://env-resource.openai.azure.com"

    def test_init_missing_api_key_raises(self):
        """Plugin raises ValueError when no API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI"):
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                    AzureOpenAIChatClient(
                        endpoint="https://test.openai.azure.com",
                        deployment_name="test",
                    )

    def test_init_missing_endpoint_raises(self):
        """Plugin raises ValueError when no endpoint."""
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "key"}, clear=True):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI"):
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                    AzureOpenAIChatClient(deployment_name="test")

    def test_init_missing_deployment_raises(self):
        """Plugin raises ValueError when no deployment name."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI"):
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                with pytest.raises(ValueError, match="deployment_name"):
                    AzureOpenAIChatClient()

    def test_init_missing_library_raises(self):
        """Plugin raises RuntimeError when openai not installed."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI", None):
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                with pytest.raises(RuntimeError, match="Install openai"):
                    AzureOpenAIChatClient(deployment_name="test")

    def test_init_with_custom_params(self):
        """Plugin accepts custom api_version, temperature, max_tokens."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                client = AzureOpenAIChatClient(
                    deployment_name="test",
                    api_version="2024-06-01",
                    temperature=0.8,
                    max_tokens=2000,
                )

                assert client.temperature == 0.8
                assert client.max_tokens == 2000
                call_kwargs = mock_azure.call_args[1]
                assert call_kwargs["api_version"] == "2024-06-01"

    def test_chat_returns_string(self):
        """chat() returns string from completion."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                mock_response = MockChatCompletion(
                    choices=[MockChoice(message=MockMessage(content="Azure response!"))]
                )
                mock_azure.return_value.chat.completions.create.return_value = mock_response

                client = AzureOpenAIChatClient(deployment_name="test")
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == "Azure response!"

    def test_chat_uses_deployment_as_model(self):
        """chat() passes deployment_name as model parameter."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                mock_response = MockChatCompletion(
                    choices=[MockChoice(message=MockMessage(content="OK"))]
                )
                mock_azure.return_value.chat.completions.create.return_value = mock_response

                client = AzureOpenAIChatClient(deployment_name="my-gpt4")
                client.chat([{"role": "user", "content": "Hi"}])

                call_kwargs = mock_azure.return_value.chat.completions.create.call_args[1]
                assert call_kwargs["model"] == "my-gpt4"

    def test_chat_empty_choices_returns_empty_string(self):
        """chat() returns empty string when no choices."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI") as mock_azure:
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                mock_response = MockChatCompletion(choices=[])
                mock_azure.return_value.chat.completions.create.return_value = mock_response

                client = AzureOpenAIChatClient(deployment_name="test")
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == ""

    def test_plugin_attributes(self):
        """Plugin has correct plugin_name and plugin_type."""
        env = {
            "AZURE_OPENAI_API_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        }
        with patch.dict("os.environ", env):
            with patch("core.llm.chat.plugins.azure_openai.AzureOpenAI"):
                from core.llm.chat.plugins.azure_openai import AzureOpenAIChatClient

                client = AzureOpenAIChatClient(deployment_name="test")

                assert client.plugin_name == "azure_openai"
                assert client.plugin_type == "chat"
