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

    Keep this intentionally minimal: this is not a “model zoo”.
    """

    max_tokens: int = 256
    temperature: float = 0.2


class LocalChatLLM:
    """
    Local ChatLLM adapter.

    Expected message format:
      - Sequence of dict-like objects with keys: 'role', 'content'
      - Roles: 'system'|'user'|'assistant' (best-effort)
    """

    def __init__(self, runtime: LocalLLMRuntime, cfg: LocalChatConfig | None = None) -> None:
        self._rt = runtime
        self._cfg = cfg or LocalChatConfig()

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        logger.info(f"{CHAT} Using local chat model (baseline quality)")

        llm = self._rt.llama()

        # Prefer OpenAI-style chat API if available in llama-cpp-python.
        if hasattr(llm, "create_chat_completion"):
            resp: Any = llm.create_chat_completion(
                messages=[{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
            )
            return _extract_chat_text(resp)

        # Fallback to plain completion with a minimal prompt format.
        prompt = _format_prompt(messages)
        resp = llm(
            prompt,
            temperature=self._cfg.temperature,
            max_tokens=self._cfg.max_tokens,
        )
        return _extract_completion_text(resp)


def _format_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts) + "\n"


def _extract_chat_text(resp: Any) -> str:
    # Best-effort across llama-cpp-python versions.
    try:
        choices = resp.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""
            return str(text)
    except Exception:
        pass
    return str(resp)


def _extract_completion_text(resp: Any) -> str:
    try:
        choices = resp.get("choices") or []
        if choices:
            return str(choices[0].get("text") or "")
    except Exception:
        pass
    return str(resp)
