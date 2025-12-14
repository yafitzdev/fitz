# tests/test_rgs_chunk_limit.py
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def test_rgs_respects_max_chunks():
    cfg = RGSConfig(max_chunks=2)
    rgs = RGS(cfg)

    chunks = [
        {"id": "1", "content": "A", "metadata": {}},
        {"id": "2", "content": "B", "metadata": {}},
        {"id": "3", "content": "C", "metadata": {}},
    ]

    prompt = rgs.build_prompt("Q?", chunks)

    assert "A" in prompt.user
    assert "B" in prompt.user
    assert "C" not in prompt.user
