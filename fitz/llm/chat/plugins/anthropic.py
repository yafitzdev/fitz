# fitz/llm/chat/plugins/anthropic.py
"""
Anthropic Claude chat plugin using centralized credentials.
"""

from __future__ import annotations

from typing import Any

from fitz.llm.credentials import CredentialError, resolve_api_key

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore


class AnthropicChatClient:
    """
    Chat plugin for Anthropic Claude API using centralized credentials.

    Required:
        - ANTHROPIC_API_KEY environment variable OR api_key parameter

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
            model: str,
            api_key: str | None = None,
            max_tokens: int = 4096,
            temperature: float = 0.2,
    ) -> None:
        if anthropic is None:
            raise RuntimeError("Install anthropic: `pip install anthropic`")

        # Use centralized credential resolution
        try:
            key = resolve_api_key(
                provider="anthropic",
                config={"api_key": api_key} if api_key else None,
            )
        except CredentialError as e:
            raise ValueError(str(e)) from e

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic(api_key=key)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response text
        """
        # Anthropic uses a separate system parameter, not a system message
        system_content = ""
        chat_messages = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            else:
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
