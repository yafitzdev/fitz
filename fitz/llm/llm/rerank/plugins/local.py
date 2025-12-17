# fitz/core/llm/rerank/plugins/local.py
from __future__ import annotations

from typing import Any, List

from fitz.backends.local_llm.embedding import LocalEmbedder, LocalEmbedderConfig
from fitz.backends.local_llm.rerank import LocalReranker, LocalRerankerConfig
from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.core.llm.rerank.base import RerankPlugin
from fitz.core.models.chunk import Chunk


class LocalRerankClient(RerankPlugin):
    plugin_name = "local"
    plugin_type = "rerank"
    availability = "local"

    def __init__(self, **kwargs: Any) -> None:
        # Extract config options
        top_k = kwargs.get("top_k", 10)

        # Create runtime
        runtime_cfg = LocalLLMRuntimeConfig(model="llama3.2:1b")
        runtime = LocalLLMRuntime(runtime_cfg)

        # Create embedder
        embedder_cfg = LocalEmbedderConfig()
        embedder = LocalEmbedder(runtime=runtime, cfg=embedder_cfg)

        # Create reranker
        reranker_cfg = LocalRerankerConfig(top_k=top_k)
        self._reranker = LocalReranker(embedder=embedder, cfg=reranker_cfg)

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank chunks using cosine similarity of embeddings.

        This will raise LLMError with friendly message if Ollama is not running.
        """
        if not chunks:
            return []

        # Extract content for reranking
        candidates = [c.content for c in chunks]

        # Get ranked indices with scores: List[Tuple[int, float]]
        ranked_indices = self._reranker.rerank(query, candidates)

        # Return chunks in ranked order
        return [chunks[idx] for idx, score in ranked_indices]
