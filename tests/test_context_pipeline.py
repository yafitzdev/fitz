# tests/test_context_pipeline.py
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.engines.classic_rag.pipeline.context.pipeline import ContextPipeline


def test_context_pipeline_end_to_end():
    chunks = [
        {"content": "A1", "file": "doc1"},
        {"content": "A2", "file": "doc1"},
        {"content": "B1", "file": "doc2"},
    ]

    out = ContextPipeline(max_chars=1000).process(chunks)

    assert isinstance(out, list)
    assert out
    assert all(isinstance(c, Chunk) for c in out)

    combined = "\n".join(c.content for c in out)
    assert "A1" in combined
    assert "A2" in combined
    assert "B1" in combined

    # Current pipeline does not propagate file->doc_id (falls back to "unknown")
    assert {c.doc_id for c in out} == {"unknown"}
