import pytest
from rag.retriever.plugins.dense import DenseRetrievalPlugin
from rag.models.chunk import Chunk


class MockEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]  # dummy vector


class MockRerankPlugin:
    def rerank(self, query, chunks):
        # Reverse list to simulate reranking
        return list(reversed(chunks))


class MockRerankEngine:
    def __init__(self):
        self.plugin = MockRerankPlugin()


class MockClient:
    def search(self, collection_name, query_vector, limit, with_payload=True):
        return [
            type("Hit", (), {"id": "1", "payload": {"text": "A"}, "score": 0.9})(),
            type("Hit", (), {"id": "2", "payload": {"text": "B"}, "score": 0.8})(),
        ]


def test_retriever_rerank_flow():
    retriever = DenseRetrievalPlugin(
        client=MockClient(),
        embed_cfg=type("Cfg", (), {"api_key": "k", "model": "m", "output_dimension": None}),
        retriever_cfg=type("Cfg", (), {"collection": "col", "top_k": 2}),
        rerank_cfg=type("Cfg", (), {"enabled": True, "api_key": "k", "model": "m"}),
        embedder=MockEmbedder(),
        rerank_engine=MockRerankEngine(),
    )

    chunks = retriever.retrieve("query")

    assert len(chunks) == 2
    assert chunks[0].text == "B"
    assert chunks[1].text == "A"
