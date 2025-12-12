from rag.retrieval.plugins.dense import DenseRetrievalPlugin
from rag.models.chunk import Chunk

class MockEmbedder:
    def embed(self, text): return [0.1, 0.2]

class MockClient:
    def search(self, collection_name, query_vector, limit, with_payload=True):
        class Hit:
            id = "123"
            score = 0.9
            payload = {"text": "hello", "file": "doc", "custom": "x"}
        return [Hit()]

def test_retriever_preserves_metadata():
    retr = DenseRetrievalPlugin(
        client=MockClient(),
        embed_cfg=type("Cfg", (), {"api_key": "k", "model": "m", "output_dimension": None}),
        retriever_cfg=type("Cfg", (), {"collection": "c", "top_k": 1}),
        embedder=MockEmbedder(),
    )

    chunks = retr.retrieve("query")
    assert isinstance(chunks[0], Chunk)

    meta = chunks[0].metadata
    assert meta["text"] == "hello"
    assert meta["file"] == "doc"
    assert meta["custom"] == "x"   # preserved
