# fitz/core/llm/embedding/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.core.llm.embedding.base import EmbeddingPlugin
from fitz.backends.local_llm.embedding import LocalEmbedder, LocalEmbedderConfig


class LocalEmbeddingClient(EmbeddingPlugin):
    """
    Local baseline EmbeddingPlugin.

    Deterministic hash embeddings.
    """

    plugin_name = "local"
    availability = "local"

    def __init__(self, **kwargs: Any) -> None:
        cfg = LocalEmbedderConfig(**kwargs)
        self._embedder = LocalEmbedder(cfg)

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed_texts([text])[0]
