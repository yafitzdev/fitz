# tests/test_anthropic_chat_plugin.py
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# Mock response structures
@dataclass
class MockTextBlock:
    text: str
    type: str = "text"


@dataclass
class MockAnthropicResponse:
    content: list[MockTextBlock]


class TestAnthropicChatPlugin:
    """Tests for Anthropic chat plugin."""

    def test_init_with_explicit_api_key(self):
        """Plugin initializes with explicit API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                client = AnthropicChatClient(api_key="test-key")

                assert client.model == "claude-sonnet-4-20250514"
                assert client.max_tokens == 4096
                assert client.temperature == 0.2
                mock_anthropic.Anthropic.assert_called_once_with(api_key="test-key")

    def test_init_with_env_var(self):
        """Plugin initializes with ANTHROPIC_API_KEY env var."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                client = AnthropicChatClient()

                mock_anthropic.Anthropic.assert_called_once_with(api_key="env-key")

    def test_init_missing_api_key_raises(self):
        """Plugin raises ValueError when no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("core.llm.chat.plugins.anthropic.anthropic"):
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    AnthropicChatClient()

    def test_init_missing_library_raises(self):
        """Plugin raises RuntimeError when anthropic not installed."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic", None):
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                with pytest.raises(RuntimeError, match="Install anthropic"):
                    AnthropicChatClient()

    def test_init_with_custom_params(self):
        """Plugin accepts custom model, max_tokens, temperature."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic"):
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                client = AnthropicChatClient(
                    model="claude-opus-4-20250514",
                    max_tokens=8192,
                    temperature=0.5,
                )

                assert client.model == "claude-opus-4-20250514"
                assert client.max_tokens == 8192
                assert client.temperature == 0.5

    def test_chat_returns_string(self):
        """chat() returns string from response."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                mock_response = MockAnthropicResponse(
                    content=[MockTextBlock(text="Hello from Claude!")]
                )
                mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

                client = AnthropicChatClient()
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == "Hello from Claude!"
                assert isinstance(result, str)

    def test_chat_handles_system_message(self):
        """chat() extracts system message and passes separately."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                mock_response = MockAnthropicResponse(
                    content=[MockTextBlock(text="Response")]
                )
                mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

                client = AnthropicChatClient()
                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                ]
                client.chat(messages)

                call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
                assert call_kwargs["system"] == "You are helpful."
                assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    def test_chat_without_system_message(self):
        """chat() works without system message."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                mock_response = MockAnthropicResponse(
                    content=[MockTextBlock(text="Response")]
                )
                mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

                client = AnthropicChatClient()
                messages = [{"role": "user", "content": "Hello"}]
                client.chat(messages)

                call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
                assert "system" not in call_kwargs

    def test_chat_multiple_content_blocks(self):
        """chat() concatenates multiple text blocks."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                mock_response = MockAnthropicResponse(
                    content=[
                        MockTextBlock(text="Part 1. "),
                        MockTextBlock(text="Part 2."),
                    ]
                )
                mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

                client = AnthropicChatClient()
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == "Part 1. Part 2."

    def test_chat_empty_content_returns_empty_string(self):
        """chat() returns empty string when no content blocks."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                mock_response = MockAnthropicResponse(content=[])
                mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

                client = AnthropicChatClient()
                result = client.chat([{"role": "user", "content": "Hi"}])

                assert result == ""

    def test_chat_preserves_conversation_order(self):
        """chat() preserves user/assistant message order."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic") as mock_anthropic:
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                mock_response = MockAnthropicResponse(
                    content=[MockTextBlock(text="Response")]
                )
                mock_anthropic.Anthropic.return_value.messages.create.return_value = mock_response

                client = AnthropicChatClient()
                messages = [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "User 1"},
                    {"role": "assistant", "content": "Assistant 1"},
                    {"role": "user", "content": "User 2"},
                ]
                client.chat(messages)

                call_kwargs = mock_anthropic.Anthropic.return_value.messages.create.call_args[1]
                assert call_kwargs["messages"] == [
                    {"role": "user", "content": "User 1"},
                    {"role": "assistant", "content": "Assistant 1"},
                    {"role": "user", "content": "User 2"},
                ]

    def test_plugin_attributes(self):
        """Plugin has correct plugin_name and plugin_type."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "key"}):
            with patch("core.llm.chat.plugins.anthropic.anthropic"):
                from core.llm.chat.plugins.anthropic import AnthropicChatClient

                client = AnthropicChatClient()

                assert client.plugin_name == "anthropic"
                assert client.plugin_type == "chat"
