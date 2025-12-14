# tests/test_retriever_engine_factory.py
from fitz.pipeline.retrieval import RetrieverEngine


class MockClient:
    def search(self, collection_name, query_vector, limit):
        return []


class MockEmbedder:
    def embed(self, text):
        return [0.0]


def test_retriever_engine_from_name():
    retriever_cfg = type("Cfg", (), {"collection": "col", "top_k": 2})

    engine = RetrieverEngine.from_name(
        "dense",
        client=MockClient(),
        retriever_cfg=retriever_cfg,
        embedder=MockEmbedder(),
        rerank_engine=None,
    )

    assert isinstance(engine, RetrieverEngine)
