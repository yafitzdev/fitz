from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os

from fitz_rag.llm.chat.base import ChatPlugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereChatClient(ChatPlugin):
    """
    Cohere chat API plugin for fitz-rag.

    - Uses the Cohere v2 Chat API (`ClientV2.chat`)
    - Accepts a list of messages (role/content)
    - Returns the assistant's text response

    This class is also auto-registered as a chat plugin via
    `plugin_name`, so it can be used through ChatEngine:

        from fitz_rag.llm.chat.engine import ChatEngine
        engine = ChatEngine.from_name("cohere")
        answer = engine.chat_text("What is RAG?")
    """

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
        """
        Run a chat completion and return the assistant's text.

        `messages` must follow the Cohere v2 format:
            [{"role": "user" | "system" | "assistant", "content": "<text>"}]
        """
        res = self._client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        # Cohere v2: response.message.content is a list of content blocks.
        # We take the first text block.
        try:
            content_blocks = res.message.content
            if not content_blocks:
                return ""
            block = content_blocks[0]
            # block is typically: {"type": "text", "text": "..."}
            text = getattr(block, "text", None) or block.get("text")  # type: ignore[attr-defined]
            return text or ""
        except Exception:
            # Fallback: try a simple str() if format changes
            return str(res)
