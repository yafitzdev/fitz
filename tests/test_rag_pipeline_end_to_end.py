# tests/test_rag_pipeline_end_to_end.py
"""
End-to-end test of RAGPipeline using mock retrieval and LLM.
"""

from dataclasses import dataclass

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSAnswer,
    RGSConfig,
)
from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline

# =============================================================================
# Mock Components
# =============================================================================


@dataclass
class MockRetrievalPipeline:
    """Mock retrieval pipeline that returns fixed chunks."""

    plugin_name: str = "mock"

    def retrieve(
        self, query: str, filter_override: dict | None = None
    ) -> list[Chunk]:
        return [
            Chunk(
                id="chunk_1",
                doc_id="doc_1",
                content="The sky is blue because of Rayleigh scattering.",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                id="chunk_2",
                doc_id="doc_2",
                content="Water appears blue due to absorption of red wavelengths.",
                chunk_index=0,
                metadata={},
            ),
        ]


class DummyLLM:
    """Mock LLM that returns a fixed response."""

    def chat(self, messages: list[dict]) -> str:
        return "The sky is blue because of Rayleigh scattering [S1]."


# =============================================================================
# Test
# =============================================================================


def test_pipeline_end_to_end():
    """
    End-to-end test of RAGPipeline:

    - uses MockRetrievalPipeline + DummyLLM
    - wires in real RGS (prompt + answer structuring)
    - asserts that an RGSAnswer is returned with sources
    """

    rgs = RGS(config=RGSConfig(max_chunks=3))

    pipe = RAGPipeline(
        retrieval=MockRetrievalPipeline(),
        chat=DummyLLM(),
        rgs=rgs,
    )

    answer = pipe.run("Why is the sky blue?")

    assert isinstance(answer, RGSAnswer)
    assert answer.answer
    assert len(answer.sources) > 0
