# tests/test_rgs_metadata_format.py
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)


def test_rgs_metadata_format():
    """Test that RGS includes chunk content in prompt."""
    rgs = RGS(RGSConfig())

    chunks = [
        {
            "id": "1",
            "content": "alpha",
            "metadata": {"file": "doc1", "a": 1, "b": 2, "c": 3},
        },
    ]

    prompt = rgs.build_prompt("Q?", chunks)

    # Content should be in the prompt
    assert "alpha" in prompt.user
    # Source label should be present
    assert "[S1]" in prompt.user
    # Query should be present
    assert "Q?" in prompt.user
