# core/llm/embedding/plugins/cohere.py
from __future__ import annotations

from dataclasses import dataclass
import os

from rag.exceptions.retriever import EmbeddingError
from core.llm.registry import register_llm_plugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereEmbeddingClient:
    plugin_name: str = "cohere"

    api_key: str | None = None
    model: str | None = None
    input_type: str | None = None
    output_dimension: int | None = None

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")

        self.model = self.model or os.getenv("COHERE_EMBED_MODEL") or "embed-english-v3.0"
        self.input_type = self.input_type or os.getenv("COHERE_EMBED_INPUT_TYPE") or "search_query"

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as exc:
            raise EmbeddingError("Failed to initialize Cohere embedding client") from exc

    def embed(self, text: str) -> list[float]:
        kwargs: dict[str, object] = {
            "texts": [text],
            "model": self.model,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension

        try:
            res = self._client.embed(**kwargs)
            return res.embeddings.float[0]
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {text!r}") from exc


register_llm_plugin(
    CohereEmbeddingClient,
    plugin_name="cohere",
    plugin_type="embedding",
)
