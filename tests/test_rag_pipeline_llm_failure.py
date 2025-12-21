# tests/test_rag_pipeline_llm_failure.py

import pytest

from fitz_ai.engines.classic_rag.exceptions import LLMError
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)
from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline


class DummyRetriever:
    def retrieve(self, q):
        return [{"id": "1", "text": "hello", "metadata": {}}]


class FailingLLM:
    def chat(self, msgs):
        raise RuntimeError("LLM is down")


def test_rag_pipeline_llm_failure():
    pipe = RAGPipeline(
        retriever=DummyRetriever(), chat=FailingLLM(), rgs=RGS(RGSConfig())
    )

    with pytest.raises(LLMError):
        pipe.run("hello?")
