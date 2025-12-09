# src/fitz_rag/llm/chat_client.py
"""
Chat client abstractions for fitz-rag.

This module defines:
- ChatClient protocol
- CohereChatClient: real chat via Cohere v2 Chat API
- DummyChatClient: trivial echo-style implementation for tests

Environment variables:
COHERE_API_KEY         # required for CohereChatClient
COHERE_CHAT_MODEL      # optional override
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional, Dict, Any
import os

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class ChatClient(Protocol):
    """Base interface for all LLM chat clients."""

    def chat(self, system_prompt: str, user_content: str) -> str:
        ...


@dataclass
class CohereChatClient:
    """
    Uses Cohere’s v2 Chat API.

    Default model changed to `command-r7b-12-2024`
    because Cohere removed `command-light` in 2025.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.3

    def __post_init__(self):
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY not set")

        # New default model (since command-light is gone)
        self.model = (
                self.model
                or os.getenv("COHERE_CHAT_MODEL")
                or "command-r7b-12-2024"
        )

        self._client = cohere.ClientV2(api_key=key)

    def _build_messages(self, system_prompt: str, user_content: str):
        msg = []
        if system_prompt:
            msg.append({"role": "system", "content": system_prompt})
        msg.append({"role": "user", "content": user_content})
        return msg

    def chat(self, system_prompt: str, user_content: str) -> str:
        messages = self._build_messages(system_prompt, user_content)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self._client.chat(**kwargs)

        # Return combined text segments
        parts = []
        for seg in response.message.content:
            if getattr(seg, "type", None) == "text":
                parts.append(seg.text)
            else:
                text = getattr(seg, "text", None)
                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts).strip()


@dataclass
class DummyChatClient:
    prefix: str = "[DUMMY-ANSWER] "

    def chat(self, system_prompt: str, user_content: str) -> str:
        return f"{self.prefix}OK"
