# fitz/llm/chat/plugins/local.py
"""
Local chat plugin using Ollama.

This plugin provides local LLM inference without API keys.
Requires Ollama to be running locally.
"""

from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.chat import LocalChatConfig, LocalChatLLM
from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.llm.chat.base import ChatPlugin


class LocalChatClient(ChatPlugin):
    """
    Local fallback chat plugin.

    Thin adapter around fitz.backends.local_llm.

    Note: This plugin ignores kwargs that don't apply to LocalChatConfig
    (like 'model', 'api_key') to allow graceful fallback from cloud configs.
    """

    plugin_name = "local"
    plugin_type = "chat"
    availability = "local"

    def __init__(self, **kwargs: Any):
        # Extract only the kwargs that LocalChatConfig accepts
        # This allows graceful fallback when config has cloud-specific kwargs
        valid_kwargs = {}
        if "max_tokens" in kwargs:
            valid_kwargs["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            valid_kwargs["temperature"] = kwargs["temperature"]

        chat_cfg = LocalChatConfig(**valid_kwargs)

        # Model can be specified in kwargs, otherwise use default
        model = kwargs.get("model", "llama3.2:1b")
        # Normalize model name (strip version suffix if needed for Ollama)
        if ":" not in model and model not in ("llama3.2:1b", "llama3.2"):
            model = "llama3.2:1b"  # Default to known working model

        runtime_cfg = LocalLLMRuntimeConfig(model=model)
        runtime = LocalLLMRuntime(runtime_cfg)

        self._llm = LocalChatLLM(runtime=runtime, cfg=chat_cfg)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        return self._llm.chat(messages)