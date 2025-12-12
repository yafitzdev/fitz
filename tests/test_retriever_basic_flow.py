import pytest
from rag.retriever.plugins.dense import RAGRetriever
from rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig
from rag.exceptions.retriever import EmbeddingError, VectorSearchError, RerankError
from rag.models.chunk import Chunk


class DummyEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class DummyQdrant:
    class Hit:
        id = "1"
        score = 0.9
        payload = {"text": "hello world", "file": "doc.txt"}

    def search(self, *args, **kwargs):
        return [DummyQdrant.Hit()]


def test_retriever_basic_flow():
    r = RAGRetriever(
        client=DummyQdrant(),
        embed_cfg=EmbeddingConfig(provider="x", model="m"),
        retriever_cfg=RetrieverConfig(collection="demo"),
        rerank_cfg=RerankConfig(enabled=False),
        embedder=DummyEmbedder(),
    )

    out = r.retrieve("hi")

    assert isinstance(out, list)
    assert isinstance(out[0], Chunk)
    assert out[0].text == "hello world"


def test_retriever_embedding_failure():
    class BadEmbedder:
        def embed(self, _): raise RuntimeError("oh no")

    r = RAGRetriever(
        client=DummyQdrant(),
        embed_cfg=EmbeddingConfig(provider="x", model="m"),
        retriever_cfg=RetrieverConfig(collection="demo"),
        embedder=BadEmbedder(),
    )

    with pytest.raises(EmbeddingError):
        r.retrieve("hi")
