# core/llm/chat/plugins/cohere.py
from __future__ import annotations

import os
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


class CohereChatClient:
    """
    Cohere chat plugin using direct HTTP requests.

    Zero dependencies beyond httpx (which is already required by fitz core).

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
        api_key: str | None = None,
        model: str = "command-r-plus",
        temperature: float = 0.2,
        base_url: str = "https://api.cohere.ai/v1",
    ) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for Cohere plugin. " "Install with: pip install httpx"
            )

        # Get API key
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it as an environment variable or pass api_key parameter."
            )
        self._api_key = key

        self.model = model
        self.temperature = temperature
        self.base_url = base_url

        # Create HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response text

        Raises:
            RuntimeError: If the API request fails
        """
        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        try:
            response = self._client.post("/chat", json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract text from response
            # Response structure varies, try common patterns
            if "text" in data:
                return data["text"]

            if "message" in data:
                msg = data["message"]
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list) and content:
                        # Handle structured content
                        text_parts = [
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ]
                        return " ".join(text_parts)

            # Fallback
            return str(data)

        except httpx.HTTPStatusError as exc:
            error_detail = ""
            try:
                error_data = exc.response.json()
                error_detail = f": {error_data.get('message', '')}"
            except Exception as e:
                # Failed to parse error response
                error_detail = f" (response parse failed: {e})"

            raise RuntimeError(
                f"Cohere API request failed with status {exc.response.status_code}{error_detail}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(f"Failed to complete chat: {exc}") from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception as e:
                # Failed to parse error response
                error_detail = f" (response parse failed: {e})"
