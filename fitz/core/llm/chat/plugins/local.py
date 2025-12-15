# fitz/core/llm/chat/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.core.llm.chat.base import ChatPlugin
from fitz.backends.local_llm.chat import LocalChatLLM, LocalChatConfig
from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig


class LocalChatClient(ChatPlugin):
    """
    Local baseline ChatPlugin.

    Purpose:
    - Zero-key fallback
    - Pipeline verification
    - Deterministic, degraded quality
    """

    plugin_name = "local"
    availability = "local"

    def __init__(self, **kwargs: Any) -> None:
        runtime_cfg = LocalLLMRuntimeConfig(**kwargs)
        runtime = LocalLLMRuntime(runtime_cfg)
        chat_cfg = LocalChatConfig()
        self._client = LocalChatLLM(runtime=runtime, cfg=chat_cfg)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        return self._client.chat(messages)
