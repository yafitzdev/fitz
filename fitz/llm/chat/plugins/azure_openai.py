# core/llm/chat/plugins/azure_openai.py
from __future__ import annotations

import os
from typing import Any

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None  # type: ignore


class AzureOpenAIChatClient:
    """
    Chat plugin for Azure OpenAI Service.

    Required environment variables:
        AZURE_OPENAI_API_KEY: API key for authentication
        AZURE_OPENAI_ENDPOINT: Azure endpoint URL

    Config example:
        llm:
          plugin_name: azure_openai
          kwargs:
            deployment_name: my-gpt4-deployment
            api_version: "2024-02-15-preview"
            temperature: 0.2
    """

    plugin_name = "azure_openai"
    plugin_type = "chat"

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment_name: str | None = None,
        api_version: str = "2024-02-15-preview",
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> None:
        if AzureOpenAI is None:
            raise RuntimeError("Install openai: `pip install openai`")

        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")

        azure_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")

        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment_name:
            raise ValueError("deployment_name is required for Azure OpenAI")

        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AzureOpenAI(
            api_key=key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
        )

    def chat(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.deployment_name,
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
