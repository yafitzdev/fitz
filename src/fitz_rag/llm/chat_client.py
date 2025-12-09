# src/fitz_rag/llm/chat_client.py
"""
Chat client abstractions for fitz-rag.

This module defines:
- ChatClient protocol
- CohereChatClient: real chat via Cohere v2 Chat API
- DummyChatClient: trivial echo-style implementation for tests

Environment variables
---------------------
COHERE_API_KEY         # required for CohereChatClient (unless api_key passed)
COHERE_CHAT_MODEL      # optional, overrides default model name
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional, Dict, Any
import os

try:  # Cohere is optional at import time
    import cohere
except ImportError:  # pragma: no cover - handled lazily at runtime
    cohere = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ChatClient protocol
# ---------------------------------------------------------------------------


class ChatClient(Protocol):
    """
    Minimal interface for chat-style LLMs.

    A chat client receives a system prompt and user content and returns
    a plain text answer.
    """

    def chat(self, system_prompt: str, user_content: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Cohere implementation
# ---------------------------------------------------------------------------


@dataclass
class CohereChatClient:
    """
    Chat client using Cohere's v2 Chat API.

    Defaults:
      - model: "command-light" (fast & cheap legacy model)
        You can override this via:
          * constructor argument `model=...`, or
          * environment variable COHERE_CHAT_MODEL.

    Note: For best quality, consider using newer models like command-r or
    command-r+ and update the default here later.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.3

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError(
                "cohere is not installed. Run `pip install cohere` inside your environment."
            )

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it in your environment or pass api_key=... to CohereChatClient."
            )

        self.model = (
            self.model
            or os.getenv("COHERE_CHAT_MODEL")
            or "command-light"
        )

        self._client = cohere.ClientV2(api_key=key)

    def _build_messages(self, system_prompt: str, user_content: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def chat(self, system_prompt: str, user_content: str) -> str:
        """
        Call the Cohere Chat API and return the assistant's text response.
        """
        messages = self._build_messages(system_prompt, user_content)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = self._client.chat(**kwargs)

        # response.message.content is a list of segments, usually with one
        # text segment. We join all text segments just in case.
        parts = []
        for segment in response.message.content:
            if getattr(segment, "type", None) == "text":
                parts.append(segment.text)
            else:
                # Fallback for plain dicts or unexpected types
                text = getattr(segment, "text", None) or getattr(segment, "content", None)
                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Dummy implementation (for tests / offline work)
# ---------------------------------------------------------------------------


@dataclass
class DummyChatClient:
    """
    Extremely simple chat client for tests.

    It just echoes back a canned response, optionally including
    parts of the input for debugging.
    """

    prefix: str = "[DUMMY-ANSWER] "

    # For debugging in tests
    last_system_prompt: Optional[str] = None
    last_user_content: Optional[str] = None

    def chat(self, system_prompt: str, user_content: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_content = user_content
        return f"{self.prefix}OK"
