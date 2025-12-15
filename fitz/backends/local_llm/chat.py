# fitz/backends/local_llm/chat.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CHAT
from fitz.backends.local_llm.runtime import LocalLLMRuntime

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalChatConfig:
    """
    Baseline local chat settings.

    Diagnostic quality only.
    """

    max_tokens: int = 256
    temperature: float = 0.2


class LocalChatLLM:
    """
    Local ChatLLM adapter using Ollama.

    NOTE:
    The Ollama Python client adapter does NOT accept keyword arguments
    like `options=` or `model=` here. The runtime already encapsulates
    the model selection.
    """

    def __init__(self, runtime: LocalLLMRuntime, cfg: LocalChatConfig | None = None) -> None:
        self._rt = runtime
        self._cfg = cfg or LocalChatConfig()

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        logger.info(f"{CHAT} Using local chat model (baseline quality)")

        llm = self._rt.llama()

        # Call adapter with messages only (no kwargs)
        resp: Any = llm.chat(
            [
                {
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                }
                for m in messages
            ]
        )

        return _extract_text(resp)


def _extract_text(resp: Any) -> str:
    try:
        msg = resp.get("message") or {}
        return str(msg.get("content") or "")
    except Exception:
        return str(resp)
