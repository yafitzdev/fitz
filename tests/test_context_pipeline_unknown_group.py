# tests/test_context_pipeline_unknown_group.py
from fitz_ai.engines.classic_rag.pipeline.context.pipeline import ContextPipeline


def test_context_pipeline_groups_unknown_file():
    chunks = [
        {"content": "A", "metadata": {}},  # no file
        {"content": "B", "file": None},  # explicit None
        {"content": "C"},  # nothing at all
    ]

    out = ContextPipeline(max_chars=200).process(chunks)

    assert out
    assert all(c.doc_id == "unknown" for c in out)
