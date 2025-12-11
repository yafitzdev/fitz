# tests/test_rgs_chunk_id_fallback.py

from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_chunk_id_fallback_generates_chunk_ids():
    rgs = RGS(RGSConfig())

    chunks = [
        {"text": "A", "metadata": {}},
        {"text": "B", "metadata": {}},
    ]

    answer = rgs.build_answer("ok", chunks)

    assert answer.sources[0].source_id == "chunk_1"
    assert answer.sources[1].source_id == "chunk_2"
