# tests/test_rgs_no_citations.py
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def test_rgs_disable_citations():
    rgs = RGS(config=RGSConfig(enable_citations=False))

    chunks = [
        {"id": "1", "content": "alpha", "metadata": {"file": "doc1"}},
        {"id": "2", "content": "beta", "metadata": {"file": "doc2"}},
    ]

    prompt = rgs.build_prompt("What?", chunks)

    assert "[S1]" not in prompt.user
    assert "[S2]" not in prompt.user
