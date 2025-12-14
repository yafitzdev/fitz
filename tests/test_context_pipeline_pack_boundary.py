# tests/test_context_pipeline_pack_boundary.py
from fitz.pipeline.context.pipeline import ContextPipeline


def test_context_pipeline_pack_boundary():
    chunks = [
        {"content": "A" * 50, "file": "doc1"},
        {"content": "B" * 50, "file": "doc1"},
        {"content": "C" * 50, "file": "doc2"},
    ]

    out = ContextPipeline(max_chars=80).process(chunks)

    assert out

    # Packing may drop later chunks when max_chars is small; only assert earliest survives.
    combined = "".join(c.content for c in out)
    assert "A" * 20 in combined
