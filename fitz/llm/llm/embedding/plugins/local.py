from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.embedding import LocalEmbedder, LocalEmbedderConfig
from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.core.llm.embedding.base import EmbeddingPlugin


class LocalEmbeddingClient(EmbeddingPlugin):
    plugin_name = "local"
    plugin_type = "embedding"
    availability = "local"

    def __init__(self, **kwargs: Any):
        embed_cfg = LocalEmbedderConfig(**kwargs)

        runtime_cfg = LocalLLMRuntimeConfig(model="llama3.2:1b")
        runtime = LocalLLMRuntime(runtime_cfg)

        self._embedder = LocalEmbedder(runtime=runtime, cfg=embed_cfg)

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)
