from fitz_rag.context.pipeline import ContextPipeline

def test_context_pipeline_groups_unknown_file():
    chunks = [
        {"text": "A", "metadata": {}},             # no file
        {"text": "B", "file": None},              # explicit None
        {"text": "C"},                            # nothing at all
    ]

    ctx = ContextPipeline(max_chars=200).build(chunks)

    # All 3 chunks must appear under "Source: unknown"
    assert "### Source: unknown" in ctx
    assert "A" in ctx
    assert "B" in ctx
    assert "C" in ctx
