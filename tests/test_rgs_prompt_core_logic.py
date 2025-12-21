# tests/test_rgs_prompt_core_logic.py
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)


def test_rgs_prompt_structure():
    rgs = RGS(RGSConfig(max_chunks=2, enable_citations=True))

    chunks = [
        {"id": "1", "content": "Hello", "metadata": {"file": "doc1"}},
        {"id": "2", "content": "World", "metadata": {"file": "doc2"}},
        {"id": "3", "content": "Ignored", "metadata": {}},
    ]

    prompt = rgs.build_prompt("What?", chunks)

    assert "You are a retrieval-grounded assistant." in prompt.system
    assert "You are given the following context snippets:" in prompt.user
    assert "[S1]" in prompt.user
    assert "[S2]" in prompt.user
    assert "Hello" in prompt.user
    assert "World" in prompt.user
    assert "Ignored" not in prompt.user
