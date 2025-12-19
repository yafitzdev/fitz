# tests/test_rgs_exclude_query.py
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSConfig


def test_rgs_excludes_query_from_context():
    rgs = RGS(config=RGSConfig(include_query_in_context=False))

    chunks = [{"id": "1", "content": "hello world", "metadata": {}}]
    prompt = rgs.build_prompt("my question", chunks)

    assert "my question" not in prompt.user
    assert "Answer the question using ONLY the context above." in prompt.user
