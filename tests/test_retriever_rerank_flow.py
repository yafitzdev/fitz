# tests/test_retriever_rerank_flow.py
from dataclasses import dataclass

from fitz.retrieval.runtime.plugins.dense import DenseRetrievalPlugin


@dataclass
class Hit:
    id: str
    payload: dict
    score: float = 0.5


class MockClient:
    def __init__(self, hits):
        self.hits = hits

    def search(self, collection_name, query_vector, limit):
        return self.hits


class MockEmbedder:
    def embed(self, text):
        return [1.0]


class MockRerankEngine:
    def rerank(self, query, chunks):
        # reverse order to prove it ran
        return list(reversed(chunks))


def test_retriever_rerank_flow():
    hits = [
        Hit(id="1", payload={"doc_id": "doc", "content": "A", "chunk_index": 0}),
        Hit(id="2", payload={"doc_id": "doc", "content": "B", "chunk_index": 1}),
    ]

    retriever_cfg = type("Cfg", (), {"collection": "col", "top_k": 2})

    retriever = DenseRetrievalPlugin(
        client=MockClient(hits),
        retriever_cfg=retriever_cfg,
        embedder=MockEmbedder(),
        rerank_engine=MockRerankEngine(),
    )

    out = retriever.retrieve("q")

    assert [c.content for c in out] == ["B", "A"]
