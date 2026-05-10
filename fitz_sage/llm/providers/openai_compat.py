# fitz_sage/llm/providers/openai_compat.py
"""
OpenAI-compatible HTTP provider wrappers.

This module implements the chat / embedding / vision providers for any
server that speaks the OpenAI HTTP protocol — OpenAI, Azure OpenAI,
llama.cpp's ``llama-server``, vLLM, LM Studio, Together, Fireworks,
Groq, OpenRouter, etc.

The classes are the *single* implementation behind the ``endpoint``,
``openai``, and ``azure_openai`` provider names. There is no legacy
"OpenAI-only" client — OpenAI itself is just one URL preset.

Authentication is delegated to the AuthProvider abstraction (see
``fitz_sage.llm.auth``):

- ``NoAuth`` for unauthenticated local servers (default for ``endpoint``).
- ``ApiKeyAuth`` for any server requiring a key (OpenAI, Together, …).
- ``M2MAuth`` / ``CompositeAuth`` for enterprise OAuth2 + API key.

Uses ``DynamicHttpxAuth`` for per-request token refresh, so M2M tokens
captured at construction time can rotate without restart.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from fitz_sage.llm.auth import AuthProvider
from fitz_sage.llm.auth.httpx_auth import DynamicHttpxAuth
from fitz_sage.llm.providers.base import ModelTier

logger = logging.getLogger(__name__)

# Default OpenAI cloud models, used when the ``openai`` preset is selected
# without an explicit model. ``endpoint`` users always specify their model.
OPENAI_CHAT_MODELS: dict[ModelTier, str] = {
    "smart": "gpt-4o",
    "balanced": "gpt-4o-mini",
    "fast": "gpt-4o-mini",
}

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_VISION_MODEL = "gpt-4o"


class OpenAICompatChat:
    """
    Chat provider for any OpenAI-compatible HTTP server.

    Args:
        auth: Authentication provider (NoAuth, ApiKeyAuth, M2MAuth, etc.).
        model: Model name. Required for ``endpoint``; defaults from
            tier table when used as the ``openai`` preset.
        tier: Model tier hint used for default-model selection
            when ``model`` is None.
        base_url: HTTP endpoint, e.g. ``http://localhost:8080/v1`` or
            ``https://api.openai.com/v1``.
        models: Optional override of the tier→model table.
        **kwargs: Passed through to ``chat.completions.create`` as defaults.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        tier: ModelTier = "smart",
        base_url: str | None = None,
        models: dict[ModelTier, str] | None = None,
        **kwargs: Any,
    ) -> None:
        import httpx
        import openai

        request_kwargs = auth.get_request_kwargs()

        http_client = httpx.Client(
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(600.0, connect=5.0),
        )

        client_kwargs: dict[str, Any] = {
            # The SDK requires a non-empty api_key string; real auth is
            # supplied by DynamicHttpxAuth on the http_client.
            "api_key": "unused",
            "http_client": http_client,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**client_kwargs)
        tier_models = models or OPENAI_CHAT_MODELS
        self._model = model or tier_models.get(tier) or OPENAI_CHAT_MODELS[tier]
        self._defaults = kwargs

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion."""
        params = {**self._defaults, **kwargs}

        response = self._client.chat.completions.create(
            model=params.pop("model", self._model),
            messages=messages,
            **params,
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or ""
        return ""

    def chat_stream(self, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[str]:
        """Generate a streaming chat completion."""
        params = {**self._defaults, **kwargs}

        stream = self._client.chat.completions.create(
            model=params.pop("model", self._model),
            messages=messages,
            stream=True,
            **params,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                if content:
                    yield content


class OpenAICompatEmbedding:
    """
    Embedding provider for any OpenAI-compatible HTTP server.

    Args:
        auth: Authentication provider.
        model: Model name (required for ``endpoint``; defaults to
            OpenAI's text-embedding-3-small for the ``openai`` preset).
        dimensions: Output dimensions (for models that support it).
        base_url: HTTP endpoint.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        import httpx
        import openai

        request_kwargs = auth.get_request_kwargs()

        http_client = httpx.Client(
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(600.0, connect=5.0),
        )

        client_kwargs: dict[str, Any] = {
            "api_key": "unused",
            "http_client": http_client,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**client_kwargs)
        self._model = model or OPENAI_EMBEDDING_MODEL
        self._dimensions = dimensions

    def embed(self, text: str, *, task_type: str | None = None) -> list[float]:
        """Embed a single text."""
        result = self.embed_batch([text])
        return result[0]

    def embed_batch(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]:
        """Embed multiple texts."""
        if not texts:
            return []

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**kwargs)

        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [list(item.embedding) for item in sorted_data]

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions."""
        if self._dimensions is None:
            try:
                result = self.embed("test")
                self._dimensions = len(result)
            except Exception:
                return 1536
        return self._dimensions or 1536


class OpenAICompatVision:
    """
    Vision provider for any OpenAI-compatible HTTP server with a
    chat-completions endpoint that accepts image content parts.

    Args:
        auth: Authentication provider.
        model: Vision-capable model name.
        base_url: HTTP endpoint.
        **kwargs: Default kwargs passed to chat.completions.create.
    """

    def __init__(
        self,
        auth: AuthProvider,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        import httpx
        import openai

        request_kwargs = auth.get_request_kwargs()

        http_client = httpx.Client(
            auth=DynamicHttpxAuth(auth),
            verify=request_kwargs.get("verify", True),
            cert=request_kwargs.get("cert"),
            timeout=httpx.Timeout(600.0, connect=5.0),
        )

        client_kwargs: dict[str, Any] = {
            "api_key": "unused",
            "http_client": http_client,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = openai.OpenAI(**client_kwargs)
        self._model = model or OPENAI_VISION_MODEL
        self._defaults = kwargs

    def describe_image(self, image_base64: str, prompt: str | None = None) -> str:
        """Describe an image using a vision-capable chat model."""
        actual_prompt = prompt or (
            "Describe this figure/chart/diagram in detail. Include any data values, "
            "labels, axes, trends, and key insights visible in the image."
        )

        # Detect image type from base64 header or default to png
        media_type = "image/png"
        if image_base64.startswith("/9j/"):
            media_type = "image/jpeg"
        elif image_base64.startswith("iVBOR"):
            media_type = "image/png"
        elif image_base64.startswith("R0lGOD"):
            media_type = "image/gif"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": actual_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_base64}"},
                    },
                ],
            }
        ]

        params = {**self._defaults}
        response = self._client.chat.completions.create(
            model=params.pop("model", self._model),
            messages=messages,
            **params,
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or ""
        return ""


__all__ = [
    "OpenAICompatChat",
    "OpenAICompatEmbedding",
    "OpenAICompatVision",
    "OPENAI_CHAT_MODELS",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_VISION_MODEL",
]
