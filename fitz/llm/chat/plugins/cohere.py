# fitz/llm/chat/plugins/cohere.py
"""
Cohere chat plugin using centralized HTTP client and credentials.

Required:
    - model: Chat model name (must be specified in config)
    - COHERE_API_KEY environment variable OR api_key parameter

Optional:
    - temperature: Sampling temperature (default: 0.2)
"""

from __future__ import annotations

from typing import Any

from fitz.core.http import (
    APIError,
    HTTPClientNotAvailable,
    create_api_client,
    handle_api_error,
    raise_for_status,
)
from fitz.llm.credentials import CredentialError, resolve_api_key


class CohereChatClient:
    """
    Cohere chat plugin using centralized HTTP client and credentials.

    Required:
        - COHERE_API_KEY environment variable OR api_key parameter

    Optional:
        - model: Chat model (default: command-r-plus)
        - temperature: Sampling temperature (default: 0.2)
    """

    plugin_name = "cohere"
    plugin_type = "chat"

    def __init__(
            self,
            model: str,
            api_key: str | None = None,
            temperature: float = 0.2,
            base_url: str = "https://api.cohere.ai/v1",
    ) -> None:
        # Use centralized credential resolution
        try:
            key = resolve_api_key(
                provider="cohere",
                config={"api_key": api_key} if api_key else None,
            )
        except CredentialError as e:
            raise RuntimeError(str(e)) from e

        self.model = model
        self.temperature = temperature
        self.base_url = base_url

        # Create HTTP client using centralized factory
        try:
            self._client = create_api_client(
                base_url=self.base_url,
                api_key=key,
                timeout_type="chat",  # 120s timeout for LLM generation
            )
        except HTTPClientNotAvailable:
            raise RuntimeError(
                "httpx is required for Cohere plugin. " "Install with: pip install httpx"
            )

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Supports: system, user, assistant roles

        Returns:
            The assistant's response text

        Raises:
            RuntimeError: If the API request fails
        """
        # Convert standard messages format to Cohere format
        preamble = ""
        chat_history = []
        current_message = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                preamble = content
            elif role == "user":
                if current_message:
                    chat_history.append({"role": "USER", "message": current_message})
                current_message = content
            elif role == "assistant":
                chat_history.append({"role": "CHATBOT", "message": content})

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "message": current_message,
            "temperature": self.temperature,
        }

        if preamble:
            payload["preamble"] = preamble

        if chat_history:
            payload["chat_history"] = chat_history

        try:
            response = self._client.post("/chat", json=payload)
            raise_for_status(response, provider="cohere", endpoint="/chat")

            data = response.json()

            if "text" in data:
                return data["text"]

            return str(data)

        except APIError as exc:
            raise RuntimeError(str(exc)) from exc

        except Exception as exc:
            error = handle_api_error(exc, provider="cohere", endpoint="/chat")
            raise RuntimeError(str(error)) from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass
