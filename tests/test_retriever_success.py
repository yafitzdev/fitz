# tests/test_retriever_success.py
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


def test_retriever_success():
    hits = [
        Hit(
            id="1", payload={"doc_id": "doc1", "content": "A", "chunk_index": 0, "x": 1}, score=0.9
        ),
        Hit(
            id="2", payload={"doc_id": "doc2", "content": "B", "chunk_index": 1, "y": 2}, score=0.8
        ),
    ]

    retriever_cfg = type("Cfg", (), {"collection": "col", "top_k": 2})

    retriever = DenseRetrievalPlugin(
        client=MockClient(hits),
        retriever_cfg=retriever_cfg,
        embedder=MockEmbedder(),
    )

    out = retriever.retrieve("q")

    assert len(out) == 2
    assert out[0].doc_id == "doc1"
    assert out[0].content == "A"
    assert out[1].doc_id == "doc2"
    assert out[1].content == "B"
