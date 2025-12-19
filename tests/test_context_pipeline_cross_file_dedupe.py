# tests/test_context_pipeline_cross_file_dedupe.py
from fitz_ai.engines.classic_rag.pipeline.context.pipeline import ContextPipeline


def test_context_pipeline_cross_file_dedupe():
    chunks = [
        {"content": "Hello WORLD", "file": "doc1"},
        {"content": "  hello   world ", "file": "doc2"},
        {"content": "HELLO WORLD ", "file": "doc3"},
    ]

    out = ContextPipeline(max_chars=500).process(chunks)

    # No cross-file dedupe; also doc_id currently falls back to "unknown"
    assert len(out) == 3
    assert [c.doc_id for c in out] == ["unknown", "unknown", "unknown"]
    assert [c.content for c in out] == ["Hello WORLD", "  hello   world ", "HELLO WORLD "]
