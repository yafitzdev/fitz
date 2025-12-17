# core/llm/chat/engine.py
from __future__ import annotations

from typing import Any

from fitz.core.llm.chat.base import ChatPlugin


class ChatEngine:
    """
    Thin wrapper around a chat plugin.

    Architecture:
    - plugin construction is done upstream (pipeline wiring)
    - engine only enforces the contract and delegates calls
    """

    def __init__(self, plugin: ChatPlugin):
        self._plugin = plugin

    @property
    def plugin(self) -> ChatPlugin:
        return self._plugin

    def chat(self, messages: list[dict[str, Any]]) -> str:
        out = self._plugin.chat(messages)
        if not isinstance(out, str):
            raise TypeError("ChatPlugin.chat must return str")
        return out
