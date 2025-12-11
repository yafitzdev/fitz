# tests/test_rgs_metadata_format.py

from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_metadata_format():
    rgs = RGS(RGSConfig())

    chunks = [
        {"id": "1", "text": "alpha", "metadata": {"file": "doc1", "a": 1, "b": 2, "c": 3}},
    ]

    prompt = rgs.build_prompt("Q?", chunks)

    # Should display file,a,b only (max_items=3)
    assert "(metadata: file='doc1', a=1, b=2, ...)" in prompt.user
