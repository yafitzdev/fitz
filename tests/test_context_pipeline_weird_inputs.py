# tests/test_context_pipeline_weird_inputs.py
from fitz.pipeline.context.pipeline import ContextPipeline


class Obj:
    def __init__(self, text: str, file: str):
        self.text = text
        self.file = file


def test_context_pipeline_weird_inputs():
    chunks = [
        Obj("alpha", "x.txt"),
        Obj("beta", "y.txt"),
    ]

    out = ContextPipeline(max_chars=200).process(chunks)

    # Current pipeline converts unknown objects into an empty unknown chunk.
    assert len(out) == 1
    assert out[0].doc_id == "unknown"
    assert out[0].content == ""
