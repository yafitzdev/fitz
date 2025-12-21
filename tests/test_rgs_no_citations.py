# tests/test_rgs_no_citations.py
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)


def test_rgs_disable_citations():
    """Test that enable_citations=False removes citation instruction from system prompt."""
    rgs = RGS(config=RGSConfig(enable_citations=False))

    chunks = [
        {"id": "1", "content": "alpha", "metadata": {"file": "doc1"}},
        {"id": "2", "content": "beta", "metadata": {"file": "doc2"}},
    ]

    prompt = rgs.build_prompt("What?", chunks)

    # Citation instruction should NOT be in system prompt
    assert "Use citations" not in prompt.system

    # Content should still be in user prompt
    assert "alpha" in prompt.user
    assert "beta" in prompt.user
