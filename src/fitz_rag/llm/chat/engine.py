from __future__ import annotations

from typing import List, Dict

from fitz_rag.exceptions.llm import LLMError
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import CHAT

logger = get_logger(__name__)


class ChatEngine:
    """
    Wraps a chat plugin and ensures:
    - consistent message formatting
    - unified error handling
    - unified logging
    """

    def __init__(self, plugin):
        self.plugin = plugin

    def chat(self, system_prompt: str, user_content: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        logger.debug(f"{CHAT} Sending {len(messages)} messages to chat provider")

        try:
            return self.plugin.chat(messages)
        except Exception as e:
            logger.error(f"{CHAT} Chat plugin failed: {e}")
            raise LLMError("Chat request failed") from e
