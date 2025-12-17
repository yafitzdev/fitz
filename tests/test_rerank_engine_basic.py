# tests/test_rerank_engine_basic.py
from fitz.llm.rerank.engine import RerankEngine
from fitz.engines.classic_rag.models.chunk import Chunk


class DummyRerankPlugin:
    def rerank(self, query, chunks):
        return list(reversed(chunks))


def test_rerank_engine_basic_flow():
    engine = RerankEngine(DummyRerankPlugin())

    chunks = [
        Chunk(id="1", doc_id="d", chunk_index=0, content="A", metadata={}),
        Chunk(id="2", doc_id="d", chunk_index=1, content="B", metadata={}),
        Chunk(id="3", doc_id="d", chunk_index=2, content="C", metadata={}),
    ]

    ranked = engine.rerank("q", chunks)

    assert [c.content for c in ranked] == ["C", "B", "A"]
