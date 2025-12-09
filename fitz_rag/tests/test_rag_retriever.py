def test_rag_retriever_import():
    pass

def test_rag_retriever_dummy(monkeypatch):
    from fitz_rag.retriever.qdrant_client import create_qdrant_client
    from fitz_rag.retriever.rag_retriever import RAGRetriever
    from fitz_rag.llm.embedding_client import DummyEmbeddingClient

    # Create dummy embedder
    embedder = DummyEmbeddingClient(dim=10)

    # Create Qdrant client (server does NOT need to run)
    client = create_qdrant_client()

    retriever = RAGRetriever(
        client=client,
        embedder=embedder,
        collection="test_collection",
        top_k=3,
    )

    # Monkeypatch Qdrant search to avoid real server calls
    class DummyRes:
        def __init__(self):
            self.payload = {"text": "hello", "meta": "x"}
            self.score = 0.99
            self.id = "dummy"

    class DummyQueryResult:
        points = [DummyRes(), DummyRes()]

    monkeypatch.setattr(
        client,
        "query_points",
        lambda **kwargs: DummyQueryResult(),
    )

    out = retriever.retrieve("example")

    assert len(out) == 2
    assert out[0].text == "hello"
