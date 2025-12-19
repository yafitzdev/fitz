# tests/test_retriever_basic_flow.py
from dataclasses import dataclass

from fitz_ai.engines.classic_rag.retrieval.runtime.engine import RetrieverEngine


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


def test_retriever_basic_flow():
    retriever_cfg = type("Cfg", (), {"collection": "col", "top_k": 1})
    hits = [Hit(id="h", payload={"doc_id": "d", "content": "X", "chunk_index": 0})]

    engine = RetrieverEngine.from_name(
        "dense",
        client=MockClient(hits),
        retriever_cfg=retriever_cfg,
        embedder=MockEmbedder(),
        rerank_engine=None,
    )

    out = engine.retrieve("q")

    assert len(out) == 1
    assert out[0].content == "X"
