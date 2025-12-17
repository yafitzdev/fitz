from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fitz.engines.classic_rag.errors.llm import LLMError
from fitz.logging.logger import get_logger
from fitz.logging.tags import PIPELINE

logger = get_logger(__name__)


_LOCAL_FALLBACK_HELP = """
===============================================
Local LLM fallback is enabled, but no local runtime was found.

fitz can run fully offline using a tiny local model via Ollama.
This is ONLY to verify that the system works. Quality does not matter.

What you need to do (one-time setup):

1) Install Ollama
   Open this link in your browser and install it:
   https://ollama.com

2) Download the tiny default model
   Open a SYSTEM TERMINAL (not Python, not PyCharm) and run:

     ollama pull llama3.2:1b

   Where to run this command:
   - Windows: Command Prompt or PowerShell
   - macOS: Terminal
   - Linux: Terminal

3) Make sure Ollama is running
   (It usually starts automatically after installation.)

Then rerun the same command you just ran.
===============================================
""".strip()


@dataclass(frozen=True)
class LocalLLMRuntimeConfig:
    """
    Local runtime config.

    This runtime is intentionally minimal and diagnostic-only.
    """

    model: str = "llama3.2:1b"
    verbose: bool = False


class _OllamaAdapter:
    """
    Thin wrapper around the Ollama Python client.

    No availability probing.
    We rely on real execution only.
    """

    def __init__(self, model: str, verbose: bool) -> None:
        try:
            import ollama  # type: ignore
        except Exception:
            raise LLMError(_LOCAL_FALLBACK_HELP) from None

        self._ollama = ollama
        self._model = model
        self._verbose = verbose

    def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            resp = self._ollama.chat(
                model=self._model,
                messages=messages,
                options={"temperature": 0.2},
            )
            return resp["message"]["content"]
        except Exception:
            raise LLMError(_LOCAL_FALLBACK_HELP) from None

    def embed(self, text: str) -> list[float]:
        try:
            resp = self._ollama.embeddings(
                model=self._model,
                prompt=text,
            )
            return resp["embedding"]
        except Exception:
            raise LLMError(_LOCAL_FALLBACK_HELP) from None


class LocalLLMRuntime:
    """
    Owns the local LLM runtime lifecycle.

    Important:
    - NO availability checks
    - NO networking probes
    - Real execution only
    """

    def __init__(self, cfg: LocalLLMRuntimeConfig | None = None) -> None:
        self._cfg = cfg or LocalLLMRuntimeConfig()
        self._adapter: Optional[_OllamaAdapter] = None

    def llama(self) -> _OllamaAdapter:
        """
        Returns a ready-to-use local LLM adapter.

        Any failure results in a clean, user-facing message.
        """
        try:
            if self._adapter is None:
                logger.info(f"{PIPELINE} Initializing local LLM runtime via Ollama")
                self._adapter = _OllamaAdapter(
                    model=self._cfg.model,
                    verbose=self._cfg.verbose,
                )
            return self._adapter
        except LLMError:
            raise
        except Exception:
            raise LLMError(_LOCAL_FALLBACK_HELP) from None
