# tests/test_rag_pipeline_end_to_end.py
from __future__ import annotations

from fitz.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSAnswer, RGSConfig
from fitz.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline


class DummyRetriever:
    """
    Minimal retrieval for pipeline tests.

    Returns a couple of dict-based chunks compatible with RGS.
    """

    def retrieve(self, query: str):
        return [
            {
                "id": "c1",
                "text": "Paris is the capital of France.",
                "metadata": {"file": "wiki_paris.txt"},
            },
            {
                "id": "c2",
                "text": "Berlin is the capital of Germany.",
                "metadata": {"file": "wiki_berlin.txt"},
            },
        ]


class DummyLLM:
    """
    Minimal LLM stub that just returns a canned answer.
    """

    def __init__(self) -> None:
        self.last_messages = None

    def chat(self, messages):
        # Keep for debugging if needed
        self.last_messages = messages
        # Pretend we used the context and answer the question
        return "Paris is the capital of France. [S1]"


def test_pipeline_end_to_end():
    """
    End-to-end test of RAGPipeline:

    - uses DummyRetriever + DummyLLM
    - wires in real RGS (prompt + answer structuring)
    - asserts that an RGSAnswer is returned with sources
    """

    rgs = RGS(config=RGSConfig(max_chunks=3))

    pipe = RAGPipeline(
        retriever=DummyRetriever(),
        chat=DummyLLM(),
        rgs=rgs,
    )

    answer = pipe.run("What is the capital of France?")

    # Basic shape
    assert isinstance(answer, RGSAnswer)
    assert isinstance(answer.answer, str)
    assert answer.answer  # non-empty text

    # We should have at least one source
    assert answer.sources
    # The first source should reference one of our chunk IDs
    ids = {s.source_id for s in answer.sources}
    assert "c1" in ids or "c2" in ids
