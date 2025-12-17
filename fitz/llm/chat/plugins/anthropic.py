# core/llm/chat/plugins/anthropic.py
from __future__ import annotations

import os
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class AnthropicChatClient:
    """
    Chat plugin for Anthropic Claude API.

    Required environment variables:
        ANTHROPIC_API_KEY: API key for authentication

    Config example:
        llm:
          plugin_name: anthropic
          kwargs:
            model: claude-sonnet-4-20250514
            max_tokens: 4096
            temperature: 0.2
    """

    plugin_name = "anthropic"
    plugin_type = "chat"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> None:
        if anthropic is None:
            raise RuntimeError("Install anthropic: `pip install anthropic`")

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set for AnthropicChatClient")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic(api_key=key)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        # Anthropic uses a separate system parameter, not a system message
        system_content = ""
        chat_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
                # Map 'assistant' stays 'assistant', 'user' stays 'user'
                chat_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": chat_messages,
        }

        if system_content:
            kwargs["system"] = system_content

        response = self._client.messages.create(**kwargs)

        # Extract text from content blocks
        if response.content:
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "".join(text_parts)

        return ""
