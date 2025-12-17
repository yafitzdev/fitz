# fitz/llm/embedding/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.embedding import LocalEmbedder, LocalEmbedderConfig
from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.llm.embedding.base import EmbeddingPlugin


class LocalEmbeddingClient(EmbeddingPlugin):
    plugin_name = "local"
    plugin_type = "embedding"
    availability = "local"

    def __init__(self, **kwargs: Any):
        # Extract model name if provided (for Ollama model selection)
        model_name = kwargs.pop("model", "nomic-embed-text")

        # LocalEmbedderConfig doesn't take any arguments currently
        embed_cfg = LocalEmbedderConfig()

        # Use the model name for the runtime
        runtime_cfg = LocalLLMRuntimeConfig(model=model_name)
        runtime = LocalLLMRuntime(runtime_cfg)

        self._embedder = LocalEmbedder(runtime=runtime, cfg=embed_cfg)
        self._model = model_name

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    @property
    def model(self) -> str:
        """Return the model name being used."""
        return self._model
