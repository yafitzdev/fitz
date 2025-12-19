# fitz/backends/local_llm/embedding.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from fitz_ai.backends.local_llm.runtime import LocalLLMRuntime
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import EMBEDDING

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalEmbedderConfig:
    """
    Baseline local embedding settings.

    Diagnostic quality only.
    """

    pass


class LocalEmbedder:
    """
    Local embedding adapter using the Ollama runtime.

    IMPORTANT:
    The Ollama Python adapter does NOT expose `embeddings()`.
    It exposes a single `embed(text: str)` method.
    """

    def __init__(self, runtime: LocalLLMRuntime, cfg: LocalEmbedderConfig | None = None) -> None:
        self._rt = runtime
        self._cfg = cfg or LocalEmbedderConfig()

    def embed(self, text: str) -> List[float]:
        logger.info(f"{EMBEDDING} Using local embedding model (baseline quality)")

        llm = self._rt.llama()

        # Correct Ollama adapter call
        vec: Any = llm.embed(text)

        if not isinstance(vec, list):
            raise TypeError("Local embedding must return list[float]")

        return vec

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        NOTE: This calls embed() for each text sequentially.
        Ollama doesn't support batch embedding via the Python client.
        """
        return [self.embed(text) for text in texts]
