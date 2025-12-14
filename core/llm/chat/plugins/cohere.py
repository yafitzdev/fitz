# core/llm/chat/plugins/cohere.py
from __future__ import annotations

from typing import Any
import os

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereChatClient:
    plugin_name = "cohere"
    plugin_type = "chat"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "command-r-plus",
        temperature: float = 0.2,
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is not set for CohereChatClient")

        self.model = model
        self.temperature = temperature
        self._client = cohere.ClientV2(api_key=key)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        msg = getattr(resp, "message", None)
        if msg is None:
            return str(resp)

        content = getattr(msg, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text

        text = getattr(msg, "text", None)
        if isinstance(text, str):
            return text

        return str(resp)
