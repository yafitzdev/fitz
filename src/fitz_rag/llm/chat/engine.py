from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from fitz_rag.llm.chat.base import ChatPlugin
from fitz_rag.llm.chat.registry import get_chat_plugin


@dataclass
class ChatEngine:
    """
    Thin orchestration layer around a chat plugin.

    Responsibilities:
    - Hold a concrete chat plugin
    - Provide a simple `.chat()` API
    - Support plugin construction by name

    Usage patterns:

        # Direct plugin use
        from fitz_rag.llm.chat.plugins.cohere import CohereChatClient
        engine = ChatEngine(CohereChatClient())
        answer = engine.chat_text("What is RAG?")

        # Plugin by name
        engine = ChatEngine.from_name("cohere")
        answer = engine.chat_text("What is RAG?")
    """

    plugin: ChatPlugin

    @classmethod
    def from_name(cls, name: str, **plugin_kwargs: Any) -> "ChatEngine":
        plugin_cls = get_chat_plugin(name)
        plugin = plugin_cls(**plugin_kwargs)  # type: ignore[arg-type]
        return cls(plugin=plugin)

    # ---------------------------------------------------------
    # Low-level API: messages
    # ---------------------------------------------------------
    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        Call the underlying plugin with a list of messages.
        """
        return self.plugin.chat(messages)

    # ---------------------------------------------------------
    # Convenience API: single user prompt
    # ---------------------------------------------------------
    def chat_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Convenience wrapper to call chat() with a single user message,
        optionally preceded by a system message.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(messages)
