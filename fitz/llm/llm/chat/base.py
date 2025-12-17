# core/llm/chat/base.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatPlugin(Protocol):
    plugin_name: str
    plugin_type: str  # must be "chat"

    def chat(self, messages: list[dict[str, Any]]) -> str: ...
