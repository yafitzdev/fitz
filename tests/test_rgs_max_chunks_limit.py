# tests/test_rgs_max_chunks_limit.py

from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_max_chunks_limit():
    rgs = RGS(RGSConfig(max_chunks=1))

    chunks = [
        {"id": "1", "text": "alpha", "metadata": {}},
        {"id": "2", "text": "beta", "metadata": {}},
    ]

    prompt = rgs.build_prompt("?", chunks)
    answer = rgs.build_answer("ok", chunks)

    # only 1 chunk allowed
    assert "alpha" in prompt.user
    assert "beta" not in prompt.user

    assert len(answer.sources) == 1
    assert answer.sources[0].source_id == "1"
