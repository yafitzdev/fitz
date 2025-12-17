# tests/test_embedding_engine_basic.py

from fitz.llm.embedding.engine import EmbeddingEngine


class DummyEmbedPlugin:
    def embed(self, text: str):
        return [1.0, 2.0, 3.0]


def test_embedding_engine_basic_call():
    engine = EmbeddingEngine(DummyEmbedPlugin())
    vec = engine.embed("hello")

    assert vec == [1.0, 2.0, 3.0]
