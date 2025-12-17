# core/llm/chat/registry.py
from __future__ import annotations

from typing import Type

from fitz.core.llm.chat.base import ChatPlugin
from fitz.core.llm.registry import get_llm_plugin


def get_chat_plugin(plugin_name: str) -> Type[ChatPlugin]:
    """
    Return the chat plugin class for the given plugin name.

    This is a thin type-safe alias over the central LLM registry.
    """
    return get_llm_plugin(plugin_name=plugin_name, plugin_type="chat")  # type: ignore[return-value]
