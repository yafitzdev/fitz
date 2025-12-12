from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os

from rag.exceptions.retriever import EmbeddingError
from core.llm.embedding.base import EmbeddingPlugin
from core.llm.registry import register_llm_plugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereEmbeddingClient(EmbeddingPlugin):
    plugin_name: str = "cohere"

    api_key: Optional[str] = None
    model: Optional[str] = None
    input_type: Optional[str] = None
    output_dimension: Optional[int] = None

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")

        self.model = self.model or os.getenv("COHERE_EMBED_MODEL") or "embed-english-v3.0"
        self.input_type = (
            self.input_type
            or os.getenv("COHERE_EMBED_INPUT_TYPE")
            or "search_query"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise EmbeddingError("Failed to initialize Cohere embedding client") from e

    def embed(self, text: str) -> List[float]:
        kwargs = {
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
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {text!r}") from e


# Register plugin on import
register_llm_plugin(
    CohereEmbeddingClient,
    plugin_name="cohere",
    plugin_type="embedding",
)
