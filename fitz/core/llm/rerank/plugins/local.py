# fitz/core/llm/rerank/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.core.llm.rerank.base import RerankPlugin
from fitz.core.models.chunk import Chunk
from fitz.backends.local_llm.embedding import LocalEmbedder, LocalEmbedderConfig
from fitz.backends.local_llm.rerank import LocalReranker, LocalRerankerConfig


class LocalRerankClient(RerankPlugin):
    """
    Local baseline RerankPlugin.

    Uses cosine similarity over local deterministic embeddings.
    """

    plugin_name = "local"
    availability = "local"

    def __init__(self, **kwargs: Any) -> None:
        embed_cfg = LocalEmbedderConfig()
        rerank_cfg = LocalRerankerConfig(**kwargs)
        embedder = LocalEmbedder(embed_cfg)
        self._reranker = LocalReranker(embedder, rerank_cfg)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        scored = self._reranker.rerank(query, texts)

        # scored = list[(index, score)] already sorted
        return [chunks[i] for i, _ in scored]
