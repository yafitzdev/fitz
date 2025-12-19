# tests/test_context_pipeline_ordering.py
from fitz_ai.engines.classic_rag.pipeline.context.pipeline import ContextPipeline


def test_context_pipeline_preserves_document_order():
    chunks = [
        {"content": "A1", "file": "doc1"},
        {"content": "B1", "file": "doc2"},
        {"content": "C1", "file": "doc3"},
        {"content": "A2", "file": "doc1"},
    ]

    out = ContextPipeline(max_chars=500).process(chunks)

    assert out
    combined = "\n".join(c.content for c in out)
    # Preserve relative ordering in the emitted context (even if packed/truncated)
    assert combined.find("A1") < combined.find("B1") < combined.find("C1")
