from __future__ import annotations

from rag.retrieval.plugins.dense import DenseRetrievalPlugin
from rag.models.chunk import Chunk

from rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text: str):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class DummyHit:
    def __init__(self, id, text, score=0.5):
        self.id = id
        self.payload = {"text": text, "file": "dummy.txt"}
        self.score = score


class DummyClient:
    def __init__(self):
        self.last_query = None

    def search(self, collection_name, query_vector, limit, with_payload=True):
        self.last_query = (collection_name, query_vector, limit)
        return [
            DummyHit("h1", "Hello world"),
            DummyHit("h2", "Another doc"),
        ]


def test_dense_retriever_calls_embed_and_search():
    embed_cfg = EmbeddingConfig(provider="x", model="y")
    retr_cfg = RetrieverConfig(collection="testcol", top_k=2)
    rerank_cfg = RerankConfig(enabled=False)

    embedder = DummyEmbedder()
    client = DummyClient()

    retriever = DenseRetrievalPlugin(
        client=client,
        embed_cfg=embed_cfg,
        retriever_cfg=retr_cfg,
        rerank_cfg=rerank_cfg,
        embedder=embedder,          # inject
        rerank_engine=None,         # ensure no rerank
    )

    chunks = retriever.retrieve("hello world")

    # Embedder called
    assert embedder.calls == ["hello world"]

    # Vector search was executed
    name, vec, limit = client.last_query
    assert name == "testcol"
    assert vec == [0.1, 0.2, 0.3]
    assert limit == 2

    # Output converted to Chunk objects
    assert len(chunks) == 2
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "Hello world"
