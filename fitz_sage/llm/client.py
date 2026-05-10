# fitz_sage/llm/client.py
"""
Public API for LLM providers.

The single chat-protocol implementation behind these factories is
``OpenAICompatChat`` (see ``fitz_sage.llm.providers.openai_compat``).
Provider names are configuration presets:

- ``endpoint/<model>`` — bring your own OpenAI-compatible URL.
- ``openai`` / ``openai/<model>`` — preset for the public OpenAI API.
- ``azure_openai/<deployment>`` — preset for tenant-specific Azure
  (requires ``base_url``).
- ``enterprise/<model>`` — separate path with OAuth2 + API key.

fitz-sage uses no embeddings, so there is no ``get_embedder``.
"""

from __future__ import annotations

from typing import Any

from fitz_sage.llm.config import (
    create_chat_provider,
    create_rerank_provider,
    create_vision_provider,
)
from fitz_sage.llm.providers.base import (
    ChatProvider,
    ModelTier,
    RerankProvider,
    VisionProvider,
)


def get_chat(
    spec: str,
    tier: ModelTier = "smart",
    config: dict[str, Any] | None = None,
) -> ChatProvider:
    """
    Get a chat provider.

    Args:
        spec: Provider spec — ``endpoint/<model>``, ``openai``,
            ``openai/<model>``, ``azure_openai/<deployment>``,
            or ``enterprise/<model>``.
        tier: Model tier (smart, balanced, fast). Used as a default-
            model hint for the ``openai`` preset; ignored when an
            explicit model is supplied.
        config: Optional config — ``base_url``, ``auth`` block, etc.

    Examples:
        >>> chat = get_chat("endpoint/qwen2.5-7b",
        ...                 config={"base_url": "http://localhost:8080/v1"})
        >>> chat = get_chat("openai/gpt-4o")
        >>> response = chat.chat([{"role": "user", "content": "Hello"}])
    """
    return create_chat_provider(spec, config, tier)


def get_reranker(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> RerankProvider | None:
    """
    Get a rerank provider.

    There is no first-class rerank backend in fitz-sage right now —
    rerank is moving to an LLM-rerank step using the chat model.
    Pass ``None`` to disable; any other spec raises with an
    actionable migration message.
    """
    return create_rerank_provider(spec, config)


def get_vision(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> VisionProvider | None:
    """
    Get a vision provider.

    Vision uses an OpenAI-compatible chat-completions endpoint with
    image content parts; any vision-capable model behind an
    ``endpoint`` URL works.

    Examples:
        >>> vision = get_vision("endpoint/qwen2-vl-7b",
        ...                     config={"base_url": "http://localhost:8080/v1"})
        >>> if vision:
        ...     description = vision.describe_image(base64_data)
    """
    return create_vision_provider(spec, config)


__all__ = [
    "get_chat",
    "get_reranker",
    "get_vision",
]
