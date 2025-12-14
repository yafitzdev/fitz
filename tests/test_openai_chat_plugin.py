# tests/test_openai_chat_plugin.py
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


class TestOpenAIChatPlugin:
    """Tests for OpenAI chat plugin."""

    def test_init_with_explicit_api_key(self):
        """Plugin initializes with explicit API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                client = OpenAIChatClient(api_key="test-key")

                assert client.model == "gpt-4o-mini"
                assert client.temperature == 0.2
                mock_openai.assert_called_once()

    def test_init_with_env_var(self):
        """Plugin initializes with OPENAI_API_KEY env var."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                client = OpenAIChatClient()

                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["api_key"] == "env-key"

    def test_init_missing_api_key_raises(self):
        """Plugin raises ValueError when no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.chat.plugins.openai.OpenAI"):
                from core.llm.chat.plugins.openai import OpenAIChatClient

                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    OpenAIChatClient()

    def test_init_missing_library_raises(self):
        """Plugin raises RuntimeError when openai not installed."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI", None):
                from core.llm.chat.plugins.openai import OpenAIChatClient

                with pytest.raises(RuntimeError, match="Install openai"):
                    OpenAIChatClient()

    def test_init_with_custom_params(self):
        """Plugin accepts custom model, temperature, max_tokens."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI"):
                from core.llm.chat.plugins.openai import OpenAIChatClient

                client = OpenAIChatClient(
                    model="gpt-4o",
                    temperature=0.8,
                    max_tokens=1000,
                )

                assert client.model == "gpt-4o"
                assert client.temperature == 0.8
                assert client.max_tokens == 1000

    def test_init_with_base_url(self):
        """Plugin accepts custom base_url for OpenAI-compatible APIs."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                OpenAIChatClient(base_url="https://custom.api.com/v1")

                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["base_url"] == "https://custom.api.com/v1"

    def test_chat_returns_string(self):
        """chat() returns string from completion."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                mock_response = MockChatCompletion(
                    choices=[MockChoice(message=MockMessage(content="Hello!"))]
                )
                mock_openai.return_value.chat.completions.create.return_value = mock_response

                client = OpenAIChatClient()
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == "Hello!"
                assert isinstance(result, str)

    def test_chat_with_system_message(self):
        """chat() handles system messages correctly."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                mock_response = MockChatCompletion(
                    choices=[MockChoice(message=MockMessage(content="Response"))]
                )
                mock_openai.return_value.chat.completions.create.return_value = mock_response

                client = OpenAIChatClient()
                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]
                result = client.chat(messages)

                assert result == "Response"
                call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]
                assert call_kwargs["messages"] == messages

    def test_chat_empty_choices_returns_empty_string(self):
        """chat() returns empty string when no choices."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                mock_response = MockChatCompletion(choices=[])
                mock_openai.return_value.chat.completions.create.return_value = mock_response

                client = OpenAIChatClient()
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == ""

    def test_chat_none_message_returns_empty_string(self):
        """chat() handles None message gracefully."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                mock_response = MockChatCompletion(
                    choices=[MockChoice(message=None)]
                )
                mock_openai.return_value.chat.completions.create.return_value = mock_response

                client = OpenAIChatClient()
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == ""

    def test_chat_includes_max_tokens_when_set(self):
        """chat() includes max_tokens in API call when specified."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI") as mock_openai:
                from core.llm.chat.plugins.openai import OpenAIChatClient

                mock_response = MockChatCompletion(
                    choices=[MockChoice(message=MockMessage(content="OK"))]
                )
                mock_openai.return_value.chat.completions.create.return_value = mock_response

                client = OpenAIChatClient(max_tokens=500)
                client.chat([{"role": "user", "content": "Hi"}])

                call_kwargs = mock_openai.return_value.chat.completions.create.call_args[1]
                assert call_kwargs["max_tokens"] == 500

    def test_plugin_attributes(self):
        """Plugin has correct plugin_name and plugin_type."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.openai.OpenAI"):
                from core.llm.chat.plugins.openai import OpenAIChatClient

                client = OpenAIChatClient()

                assert client.plugin_name == "openai"
                assert client.plugin_type == "chat"
