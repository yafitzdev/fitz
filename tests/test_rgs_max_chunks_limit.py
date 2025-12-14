# tests/test_rgs_max_chunks_limit.py
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def test_rgs_max_chunks_limit():
    rgs = RGS(RGSConfig(max_chunks=1))

    chunks = [
        {"id": "1", "content": "alpha", "metadata": {}},
        {"id": "2", "content": "beta", "metadata": {}},
    ]

    prompt = rgs.build_prompt("?", chunks)

    assert "alpha" in prompt.user
    assert "beta" not in prompt.user
