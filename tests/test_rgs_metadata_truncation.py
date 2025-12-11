from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_metadata_truncation():
    rgs = RGS(RGSConfig())

    chunk = {
        "id": "x",
        "text": "hello",
        "metadata": {
            "a": 1,
            "b": 2,
            "c": 3,
            "d": 4,   # should be truncated
            "e": 5
        }
    }

    prompt = rgs.build_prompt("q", [chunk])

    # Only first 3 items shown + ellipsis
    assert "(metadata:" in prompt.user
    assert "a=1" in prompt.user
    assert "b=2" in prompt.user
    assert "c=3" in prompt.user
    assert "..." in prompt.user

    # Must not show items beyond 3
    assert "d=4" not in prompt.user
    assert "e=5" not in prompt.user
