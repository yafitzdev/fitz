# core/llm/embedding/plugins/azure_openai.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fitz.engines.classic_rag.errors.llm import EmbeddingError

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None  # type: ignore


@dataclass
class AzureOpenAIEmbeddingClient:
    """
    Embedding plugin for Azure OpenAI Service.

    Required environment variables:
        AZURE_OPENAI_API_KEY: API key for authentication
        AZURE_OPENAI_ENDPOINT: Azure endpoint URL

    Config example:
        embedding:
          plugin_name: azure_openai
          kwargs:
            deployment_name: my-embedding-deployment
            api_version: "2024-02-15-preview"
    """

    plugin_name: str = "azure_openai"
    plugin_type: str = "embedding"

    api_key: str | None = None
    endpoint: str | None = None
    deployment_name: str | None = None
    api_version: str = "2024-02-15-preview"
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if AzureOpenAI is None:
            raise RuntimeError("Install openai: `pip install openai`")

        key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not key:
            raise RuntimeError("AZURE_OPENAI_API_KEY is not set")

        azure_endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")

        self.deployment_name = self.deployment_name or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        )
        if not self.deployment_name:
            raise RuntimeError("deployment_name is required for Azure OpenAI embedding")

        try:
            self._client = AzureOpenAI(
                api_key=key,
                api_version=self.api_version,
                azure_endpoint=azure_endpoint,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to initialize Azure OpenAI embedding client") from exc

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {
            "input": text,
            "model": self.deployment_name,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions

        try:
            response = self._client.embeddings.create(**kwargs)
            return response.data[0].embedding
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {text[:50]!r}...") from exc
