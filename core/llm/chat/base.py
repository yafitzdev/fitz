# core/llm/chat/base.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatPlugin(Protocol):
    """
    Canonical chat plugin contract.

    Provider-specific logic must live in plugin implementations only.
    """

    def chat(self, messages: list[dict[str, Any]]) -> str:
        ...
