# tests/test_dense_retriever_basic_embedding_and_search.py
from dataclasses import dataclass

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.engines.classic_rag.retrieval.runtime.plugins.dense import DenseRetrievalPlugin


@dataclass
class Hit:
    id: str
    payload: dict
    score: float = 0.5


class MockClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, collection_name, query_vector, limit):
        self.calls.append((collection_name, query_vector, limit))
        return self.hits


class MockEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


def test_dense_retriever_calls_embed_and_search():
    hits = [Hit(id="h1", payload={"doc_id": "d1", "content": "c1", "chunk_index": 0})]
    client = MockClient(hits)

    retriever_cfg = type("Cfg", (), {"collection": "col", "top_k": 2})
    embedder = MockEmbedder()

    retr = DenseRetrievalPlugin(client=client, retriever_cfg=retriever_cfg, embedder=embedder)

    out = retr.retrieve("q")

    assert embedder.calls == ["q"]
    assert client.calls == [("col", [0.1, 0.2, 0.3], 2)]
    assert isinstance(out, list) and isinstance(out[0], Chunk)
