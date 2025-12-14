# tests/test_rgs_metadata_truncation.py
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def test_rgs_metadata_truncation():
    rgs = RGS(RGSConfig())

    chunk = {
        "id": "x",
        "content": "hello",
        "metadata": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
    }

    prompt = rgs.build_prompt("q", [chunk])

    assert "a=1" in prompt.user
    assert "b=2" in prompt.user
    assert "c=3" in prompt.user
    assert "..." in prompt.user
