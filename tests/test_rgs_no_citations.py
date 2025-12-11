import pytest
from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_disable_citations():
    cfg = RGSConfig(enable_citations=False)
    rgs = RGS(config=cfg)

    chunks = [
        {"id": "1", "text": "alpha", "metadata": {"file": "doc1"}},
        {"id": "2", "text": "beta",  "metadata": {"file": "doc2"}},
    ]

    prompt = rgs.build_prompt("What?", chunks)

    # System must not contain citation instructions
    assert "citation" not in prompt.system.lower()
    assert "[S1]" not in prompt.user
    assert "[S2]" not in prompt.user
