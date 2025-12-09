# src/fitz_rag/llm/chat_client.py
"""
Chat client abstraction for fitz-rag.

This module defines a minimal interface for LLM chat providers and
a dummy implementation for testing.

You can implement this Protocol using:
- OpenAI
- Azure OpenAI
- local models
- any other chat provider

As long as it implements: chat(system_prompt, user_content) -> str
"""

from __future__ import annotations

from typing import Protocol


class ChatClient(Protocol):
    """
    Minimal interface for chat-based LLM providers.
    """

    def chat(self, system_prompt: str, user_content: str) -> str:
        """
        Execute a single-turn chat completion with a system and user message.

        Returns the model's response as a plain string.
        """
        ...


class DummyChatClient:
    """
    Simple dummy chat client for testing and offline development.

    It just echoes back a canned response, optionally including
    parts of the input for debugging.
    """

    def __init__(self, prefix: str = "[DUMMY-ANSWER] ") -> None:
        self.prefix = prefix
        self.last_system_prompt: str | None = None
        self.last_user_content: str | None = None

    def chat(self, system_prompt: str, user_content: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_content = user_content
        # For tests, just return a deterministic message
        return f"{self.prefix}OK"
