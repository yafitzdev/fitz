def test_import_sourcer_base():
    pass


def test_prompt_builder_runs():
    from fitz_rag.sourcer.rag_base import (
        SourceConfig,
        RetrievalContext,
    )
    from fitz_rag.sourcer.prompt_builder import (
        build_user_prompt,
    )

    # Minimal context
    ctx = RetrievalContext(
        query="test query",
        artefacts={"dummy": []},
    )

    sources = [SourceConfig(name="dummy", order=1, strategy=None)]

    prompt = build_user_prompt(
        trf={"a": 1},
        ctx=ctx,
        prompt_text="Do something",
        sources=sources,
    )

    assert "TRF JSON" in prompt
    assert "RAG CONTEXT" in prompt
