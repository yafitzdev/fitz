# tests/test_rgs_chunk_id_fallback.py
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSConfig


def test_rgs_chunk_id_fallback_generates_chunk_ids():
    rgs = RGS(RGSConfig())

    chunks = [
        {"content": "alpha", "metadata": {}},  # missing id
        {"content": "beta", "metadata": {}},  # missing id
    ]

    ans = rgs.build_answer("ok", chunks)

    assert ans.sources[0].source_id == "chunk_1"
    assert ans.sources[1].source_id == "chunk_2"
