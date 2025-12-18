# fitz/llm/chat/plugins/openai.py
"""
OpenAI chat plugin using centralized credentials.
"""

from __future__ import annotations

from typing import Any

from fitz.llm.credentials import CredentialError, resolve_api_key

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class OpenAIChatClient:
    """
    Chat plugin for OpenAI API using centralized credentials.

    Required:
        - OPENAI_API_KEY environment variable OR api_key parameter

    Config example:
        llm:
          plugin_name: openai
          kwargs:
            model: gpt-4o
            temperature: 0.2
    """

    plugin_name = "openai"
    plugin_type = "chat"

    def __init__(
            self,
            model: str,
            api_key: str | None = None,
            temperature: float = 0.2,
            max_tokens: int | None = None,
            base_url: str | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("Install openai: `pip install openai`")

        # Use centralized credential resolution
        try:
            key = resolve_api_key(
                provider="openai",
                config={"api_key": api_key} if api_key else None,
            )
        except CredentialError as e:
            raise ValueError(str(e)) from e

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response text
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0] if response.choices else None
        if choice is None:
            return ""

        message = choice.message
        if message is None:
            return ""

        return message.content or ""
