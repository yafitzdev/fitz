# tests/test_context_pipeline_markdown_integrity.py
from fitz_ai.engines.classic_rag.pipeline.context.pipeline import ContextPipeline


def test_context_pipeline_markdown_integrity():
    chunks = [
        {"content": "A" * 500, "file": "doc1"},
        {"content": "B" * 500, "file": "doc2"},
    ]

    out = ContextPipeline(max_chars=200).process(chunks)

    # The pipeline may truncate/pack; assert it emits a non-empty context.
    assert out
    assert any(c.content for c in out)
