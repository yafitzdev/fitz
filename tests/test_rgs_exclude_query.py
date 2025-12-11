from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_excludes_query_from_context():
    cfg = RGSConfig(include_query_in_context=False)
    rgs = RGS(config=cfg)

    chunks = [
        {"id": "1", "text": "hello world", "metadata": {}},
    ]

    prompt = rgs.build_prompt("my question", chunks)

    # Query should not appear anywhere in the user prompt
    assert "my question" not in prompt.user
    # Should instead have the generic instructions
    assert "Answer the question using ONLY the context above." in prompt.user
