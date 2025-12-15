# fitz/backends/local_llm/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalLLMRuntimeConfig:
    """
    Local runtime config for a single ultra-light model.

    Note:
      - This backend is baseline/diagnostic quality.
      - It is intended to make the pipeline runnable without API keys.
    """

    model_path: str
    n_ctx: int = 2048
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0
    seed: int = 0
    verbose: bool = False


class LocalLLMRuntime:
    """
    Owns the local model lifecycle.

    Implementation strategy:
      - Prefer llama-cpp-python if installed.
      - If missing, callers can still use the deterministic non-LLM embed/rerank
        fallbacks, but chat will fail with a clear error.
    """

    def __init__(self, cfg: LocalLLMRuntimeConfig) -> None:
        self._cfg = cfg
        self._llama: Optional[Any] = None

    @property
    def cfg(self) -> LocalLLMRuntimeConfig:
        return self._cfg

    def is_loaded(self) -> bool:
        return self._llama is not None

    def load(self) -> None:
        if self._llama is not None:
            return

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Local chat model backend requires 'llama-cpp-python'. "
                "Install it (and provide a model_path) or configure a remote Chat LLM."
            ) from e

        logger.info(f"{PIPELINE} Loading local LLM model: {self._cfg.model_path}")

        kwargs: dict[str, Any] = {
            "model_path": self._cfg.model_path,
            "n_ctx": self._cfg.n_ctx,
            "seed": self._cfg.seed,
            "verbose": self._cfg.verbose,
            "n_gpu_layers": self._cfg.n_gpu_layers,
        }
        if self._cfg.n_threads is not None:
            kwargs["n_threads"] = self._cfg.n_threads

        self._llama = Llama(**kwargs)

    def llama(self) -> Any:
        """
        Returns the underlying llama-cpp model instance.

        Callers must not store this reference outside adapter scope.
        """
        if self._llama is None:
            self.load()
        assert self._llama is not None
        return self._llama

    def close(self) -> None:
        # llama-cpp-python doesn't require explicit close in most cases.
        # Keep a hook for symmetry and future runtimes.
        self._llama = None
