from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from core.llm.chat.base import ChatPlugin
from core.llm.registry import get_llm_plugin
from core.logging.logger import get_logger
from core.logging.tags import CHAT

logger = get_logger(__name__)


@dataclass
class ChatEngine:
    """
    Orchestration layer around a ChatPlugin.

    Responsibilities:
        - Normalize chat messages
        - Call plugin.chat(...)
        - Provide factory constructor from_name(...)
    """

    plugin: ChatPlugin

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        logger.info(f"{CHAT} Running chat with {len(messages)} messages")
        return self.plugin.chat(messages)

    # ---------------------------------------------------------
    # Factory from registry
    # ---------------------------------------------------------
    @classmethod
    def from_name(cls, plugin_name: str, **kwargs) -> "ChatEngine":
        PluginCls = get_llm_plugin(plugin_name, plugin_type="chat")
        plugin = PluginCls(**kwargs)
        return cls(plugin=plugin)
