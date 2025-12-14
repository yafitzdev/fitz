# tests/test_rgs_metadata_format.py
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def test_rgs_metadata_format():
    rgs = RGS(RGSConfig())

    chunks = [
        {"id": "1", "content": "alpha", "metadata": {"file": "doc1", "a": 1, "b": 2, "c": 3}},
    ]

    prompt = rgs.build_prompt("Q?", chunks)

    assert "metadata:" in prompt.user
    assert "file='doc1'" in prompt.user
    assert "a=1" in prompt.user
    assert "b=2" in prompt.user
    assert "..." in prompt.user
