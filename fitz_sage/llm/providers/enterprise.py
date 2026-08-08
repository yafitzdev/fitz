# fitz_sage/llm/providers/enterprise.py
"""
Enterprise gateway provider.

Simple httpx-based client for enterprise LLM gateways. No SDK dependencies.
Assumes OpenAI-compatible API format (POST /chat/completions).

Model strings are passed through verbatim - the gateway interprets them.
Examples: "openai/gpt-4o" (BMW), "gpt-4o" (generic), "my-deployment" (Azure)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from fitz_sage.llm.auth import AuthProvider
from fitz_sage.llm.auth.httpx_auth import DynamicHttpxAuth

logger = logging.getLogger(__name__)


class EnterpriseChat:
    """
    Enterprise gateway chat provider.

    Args:
        auth: Authentication provider (M2MAuth, CompositeAuth, etc.)
        base_url: Gateway URL (e.g., "https://llm.corp.internal/v1")
        model: Model string passed verbatim to gateway
        **kwargs: Default kwargs for chat calls (temperature, max_tokens, etc.)
    """

    def __init__(
        self,
        auth: AuthProvider | None,
        base_url: str,
        model: str,
        **kwargs: Any,
    ) -> None:
        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "timeout": httpx.Timeout(600.0, connect=5.0),
        }
        if auth is not None:
            request_kwargs = auth.get_request_kwargs()
            client_kwargs["auth"] = DynamicHttpxAuth(auth)
            client_kwargs["verify"] = request_kwargs.get("verify", True)
            client_kwargs["cert"] = request_kwargs.get("cert")

        self._client = httpx.Client(**client_kwargs)
        self._model = model
        self._defaults = kwargs

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion."""
        params = {**self._defaults, **kwargs}

        body = {
            "model": params.pop("model", self._model),
            "messages": messages,
            **params,
        }

        response = self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        # OpenAI-compatible response format
        if "choices" in data and data["choices"]:
            return str(data["choices"][0].get("message", {}).get("content", ""))
        return ""


__all__ = [
    "EnterpriseChat",
]
