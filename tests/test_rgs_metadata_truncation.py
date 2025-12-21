# tests/test_rgs_metadata_truncation.py
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)


def test_rgs_metadata_truncation():
    """Test that RGS handles chunks with metadata."""
    rgs = RGS(RGSConfig())

    chunk = {
        "id": "x",
        "content": "hello",
        "metadata": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
    }

    prompt = rgs.build_prompt("q", [chunk])

    # Content should be in the prompt
    assert "hello" in prompt.user
    # Query should be present
    assert "q" in prompt.user
