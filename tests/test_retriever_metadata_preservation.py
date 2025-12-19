# tests/test_retriever_metadata_preservation.py
from dataclasses import dataclass

from fitz_ai.engines.classic_rag.retrieval.runtime.plugins.dense import DenseRetrievalPlugin


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


def test_retriever_preserves_metadata():
    hits = [
        Hit(
            id="h",
            payload={"doc_id": "doc", "content": "text", "chunk_index": 7, "keep_me": 123},
            score=0.42,
        )
    ]

    retriever_cfg = type("Cfg", (), {"collection": "c", "top_k": 1})

    retr = DenseRetrievalPlugin(
        client=MockClient(hits), retriever_cfg=retriever_cfg, embedder=MockEmbedder()
    )
    out = retr.retrieve("q")

    assert out[0].metadata["keep_me"] == 123
    assert out[0].metadata["doc_id"] == "doc"
    assert "score" in out[0].metadata
