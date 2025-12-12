# core/llm/chat/engine.py
"""
Chat engine wiring for Fitz LLMs.

Responsibilities:
- Read validated config
- Resolve credentials
- Instantiate chat plugin
- Execute chat calls
"""

from __future__ import annotations

from typing import Iterable

from core.logging.logger import get_logger
from core.logging.tags import CHAT

from core.llm.credentials import resolve_api_key
from core.llm.chat.registry import get_chat_plugin
from core.config.schema import FitzConfig

logger = get_logger(__name__)


class ChatEngine:
    def __init__(self, config: FitzConfig):
        self._config = config

        llm_cfg = config.llm
        provider = llm_cfg.provider

        api_key = resolve_api_key(
            provider=provider,
            config=llm_cfg.model_dump(),
        )

        logger.info(f"{CHAT} Initializing chat engine for provider '{provider}'")

        plugin_cls = get_chat_plugin(provider)
        self._plugin = plugin_cls(
            api_key=api_key,
            model=llm_cfg.model,
        )

    def chat(self, messages: Iterable[dict[str, str]]) -> str:
        """
        Execute a chat request.

        Parameters
        ----------
        messages:
            OpenAI-style message list:
            [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns
        -------
        str
            Assistant response
        """
        return self._plugin.chat(messages)
