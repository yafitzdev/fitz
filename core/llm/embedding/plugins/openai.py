# core/llm/embedding/plugins/openai.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from core.exceptions.llm import EmbeddingError

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


@dataclass
class OpenAIEmbeddingClient:
    """
    Embedding plugin for OpenAI API.

    Required environment variables:
        OPENAI_API_KEY: API key for authentication

    Config example:
        embedding:
          plugin_name: openai
          kwargs:
            model: text-embedding-3-small
            dimensions: 1536
    """

    plugin_name: str = "openai"
    plugin_type: str = "embedding"

    api_key: str | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError("Install openai: `pip install openai`")

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client_kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        try:
            self._client = OpenAI(**client_kwargs)
        except Exception as exc:
            raise EmbeddingError("Failed to initialize OpenAI embedding client") from exc

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {
            "input": text,
            "model": self.model,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        try:
            response = self._client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {text[:50]!r}...") from exc
