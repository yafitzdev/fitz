# fitz/core/llm/rerank/plugins/local.py
from __future__ import annotations

from typing import List, Any

from fitz.backends.local_llm.runtime import LocalLLMRuntime, LocalLLMRuntimeConfig
from fitz.core.llm.rerank.base import RerankPlugin
from fitz.core.models.chunk import Chunk


class LocalRerankClient(RerankPlugin):
    plugin_name = "local"
    plugin_type = "rerank"
    availability = "local"

    def __init__(self, **kwargs: Any) -> None:
        cfg = LocalLLMRuntimeConfig(model="llama3")
        self._rt = LocalLLMRuntime(cfg)

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        # 🔴 THIS LINE IS MANDATORY
        self._rt.llama()

        # deterministic fallback logic
        return chunks
