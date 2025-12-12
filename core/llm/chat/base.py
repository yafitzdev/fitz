from __future__ import annotations

from typing import Protocol, List, Dict, Any


class ChatPlugin(Protocol):
    """
    Protocol for chat / LLM plugins.

    Any chat implementation (Cohere, OpenAI, etc.) should implement this.

    Plugins typically live in:
        fitz_rag.llm.chat.plugins.<name>

    and declare a unique:
        plugin_name: str
    """

    # plugin_name: str = "unique-name"

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run a chat completion given a list of messages.

        Messages follow the usual structure:
            {"role": "user" | "assistant" | "system", "content": "<text>"}
        and the plugin should return the assistant's response text.
        """
        ...
