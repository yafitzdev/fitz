# core/llm/chat/plugins/openai.py
from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class OpenAIChatClient:
    """
    Chat plugin for OpenAI API.

    Required environment variables:
        OPENAI_API_KEY: API key for authentication

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
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("Install openai: `pip install openai`")

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set for OpenAIChatClient")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        client_kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

    def chat(self, messages: list[dict[str, Any]]) -> str:
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
