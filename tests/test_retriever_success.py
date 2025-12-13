from rag.retrieval.plugins.dense import DenseRetrievalPlugin
from core.models.chunk import Chunk

class MockEmbedder:
    def embed(self, text):
        return [1.0, 2.0, 3.0]  # any vector

class MockHit:
    def __init__(self, id, text, score):
        self.id = id
        self.payload = {"text": text, "file": "docX"}
        self.score = score

class MockClient:
    def __init__(self):
        self.called = False

    def search(self, collection_name, query_vector, limit, with_payload=True):
        self.called = True
        return [
            MockHit(1, "A", 0.9),
            MockHit(2, "B", 0.8),
        ]

def test_retriever_success():
    retriever = DenseRetrievalPlugin(
        client=MockClient(),
        embed_cfg=None,         # We inject embedder manually
        retriever_cfg=type("Cfg", (), {"collection": "col", "top_k": 2}),
        rerank_cfg=None,
        embedder=MockEmbedder(),
    )

    chunks = retriever.retrieve("hello")

    assert len(chunks) == 2
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].text == "A"
    assert chunks[0].metadata["file"] == "docX"
    assert retriever.client.called is True
