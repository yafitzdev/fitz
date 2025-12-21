# tests/test_rag_pipeline_llm_failure.py
"""
Test that RAGPipeline properly handles LLM failures.
"""

from dataclasses import dataclass

import pytest

from fitz_ai.engines.classic_rag.exceptions import LLMError
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSConfig,
)
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline


@dataclass
class MockRetrievalPipeline:
    """Mock retrieval pipeline that returns fixed chunks."""

    plugin_name: str = "mock"

    def retrieve(self, query: str) -> list[Chunk]:
        return [
            Chunk(
                id="chunk_1",
                doc_id="doc_1",
                content="Some content.",
                chunk_index=0,
                metadata={},
            ),
        ]


class FailingLLM:
    """Mock LLM that always raises an exception."""

    def chat(self, messages: list[dict]) -> str:
        raise RuntimeError("LLM service unavailable")


def test_rag_pipeline_llm_failure():
    pipe = RAGPipeline(
        retrieval=MockRetrievalPipeline(),
        chat=FailingLLM(),
        rgs=RGS(RGSConfig()),
    )

    with pytest.raises(LLMError):
        pipe.run("Any query")