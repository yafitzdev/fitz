from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fitz.core.exceptions.llm import LLMError
from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE

logger = get_logger(__name__)


_LOCAL_FALLBACK_HELP = """\

===============================================
Local LLM fallback is enabled, but no local runtime was found.
fitz can run fully offline using a lightweight local model via Ollama.

To enable local fallback:
  1) Install Ollama: https://ollama.com
  2) Download the default model:
     ollama pull llama3
  3) Ensure Ollama is running in the background

Then rerun your command.
===============================================
"""


@dataclass(frozen=True)
class LocalLLMRuntimeConfig:
    model: str = "llama3"
    timeout: float = 1.0


class LocalLLMRuntime:
    def __init__(self, cfg: LocalLLMRuntimeConfig) -> None:
        self._cfg = cfg

    def _check_available(self) -> None:
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=self._cfg.timeout):
                return

        except Exception:
            # 🔑 THIS IS THE KEY LINE
            raise LLMError(_LOCAL_FALLBACK_HELP) from None

    def llama(self) -> Any:
        self._check_available()

        try:
            import ollama  # type: ignore
        except Exception:
            raise LLMError(_LOCAL_FALLBACK_HELP) from None

        logger.info(f"{PIPELINE} Using local Ollama model: {self._cfg.model}")

        class _OllamaAdapter:
            def __init__(self, model: str):
                self._model = model

            def chat(self, prompt: str) -> str:
                resp = ollama.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp["message"]["content"]

        return _OllamaAdapter(self._cfg.model)
