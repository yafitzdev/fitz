from fitz_rag.context.pipeline import ContextPipeline

def test_context_pipeline_end_to_end():
    chunks = [
        {"text": "A1", "metadata": {"file": "doc1"}},
        {"text": "A2", "metadata": {"file": "doc1"}},
        {"text": "B1", "metadata": {"file": "doc2"}},
    ]

    pipe = ContextPipeline(max_chars=1000)
    ctx = pipe.build(chunks)

    assert "### Source: doc1" in ctx
    assert "A1" in ctx and "A2" in ctx
    assert "### Source: doc2" in ctx
