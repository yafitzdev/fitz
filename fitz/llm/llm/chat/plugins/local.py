from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.chat import LocalChatConfig, LocalChatLLM
from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.core.llm.chat.base import ChatPlugin


class LocalChatClient(ChatPlugin):
    """
    Local fallback chat plugin.

    Thin adapter around fitz.backends.local_llm.
    """

    plugin_name = "local"
    plugin_type = "chat"
    availability = "local"

    def __init__(self, **kwargs: Any):
        chat_cfg = LocalChatConfig(**kwargs)

        runtime_cfg = LocalLLMRuntimeConfig(model="llama3.2:1b")

        runtime = LocalLLMRuntime(runtime_cfg)

        self._llm = LocalChatLLM(runtime=runtime, cfg=chat_cfg)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        return self._llm.chat(messages)
