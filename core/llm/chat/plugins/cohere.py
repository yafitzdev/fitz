# core/llm/plugins/cohere.py  (chat portion)
from __future__ import annotations

from typing import Any
import os

from core.llm.registry import register_llm_plugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereChatClient:
    plugin_name = "cohere"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "command-r-plus",
        temperature: float = 0.2,
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set for CohereChatClient")

        self._client = cohere.ClientV2(api_key=self.api_key)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        # Cohere v2 returns message content segments; normalize to a plain string.
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


register_llm_plugin(
    CohereChatClient,
    plugin_name="cohere",
    plugin_type="chat",
)
