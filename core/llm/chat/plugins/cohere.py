# ============================
# File: src/fitz_stack/llm/chat/plugins/cohere.py
# ============================
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os

from core.llm.chat.base import ChatPlugin
from core.llm.registry import register_llm_plugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereChatClient(ChatPlugin):
    plugin_name: str = "cohere"

    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.3

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")

        self.model = (
            self.model
            or os.getenv("COHERE_CHAT_MODEL")
            or "command-a-03-2025"
        )

        self._client = cohere.ClientV2(api_key=key)

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        res = self._client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        try:
            blocks = res.message.content
            if not blocks:
                return ""
            block = blocks[0]
            return getattr(block, "text", None) or block.get("text", "")
        except Exception:
            return str(res)


# Register plugin on import
register_llm_plugin(
    CohereChatClient,
    plugin_name="cohere",
    plugin_type="chat",
)
