from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from fitz_rag.llm.embedding.base import EmbeddingPlugin
from fitz_rag.llm.embedding.registry import get_embedding_plugin


@dataclass
class EmbeddingEngine:
    """
    Thin orchestration layer around an embedding plugin.

    Responsibilities:
    - Hold a concrete embedding plugin instance
    - Provide a stable `.embed(text)` API
    - Optionally construct a plugin by name via `from_name`

    Existing code typically does:

        from fitz_rag.llm.embedding.engine import EmbeddingEngine
        from fitz_rag.llm.embedding.plugins.cohere import CohereEmbeddingClient

        plugin = CohereEmbeddingClient(...)
        engine = EmbeddingEngine(plugin)
        vec = engine.embed("Hello")

    New code can also do:

        engine = EmbeddingEngine.from_name("cohere", api_key=..., model=...)
        vec = engine.embed("Hello")
    """

    plugin: EmbeddingPlugin

    @classmethod
    def from_name(cls, name: str, **plugin_kwargs: Any) -> "EmbeddingEngine":
        """
        Construct an EmbeddingEngine from a registered plugin name
        and keyword arguments required by that plugin.
        """
        plugin_cls = get_embedding_plugin(name)
        plugin = plugin_cls(**plugin_kwargs)  # type: ignore[arg-type]
        return cls(plugin=plugin)

    def embed(self, text: str) -> List[float]:
        """
        Delegate embedding to the underlying plugin.
        """
        return self.plugin.embed(text)
