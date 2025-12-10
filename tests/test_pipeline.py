from typing import List, Dict

from fitz_rag.core.types import RetrievedChunk
from fitz_rag.sourcer.rag_base import (
    ArtefactRetrievalStrategy,
    SourceConfig,
    RAGContextBuilder,
)
from fitz_rag.llm.chat_client import DummyChatClient
from fitz_rag.pipeline.engine import run_single_rag_analysis


class DummyStrategy(ArtefactRetrievalStrategy):
    def retrieve(self, trf: Dict, query: str) -> List[RetrievedChunk]:
        return [
            RetrievedChunk(
                collection="dummy",
                score=0.99,
                text=f"chunk for {query}",
                metadata={"file": "dummy.txt"},
                chunk_id="1",
            )
        ]


def test_run_single_rag_analysis():
    # 1) Setup source config + context builder
    strategy = DummyStrategy()
    src_cfg = SourceConfig(
        name="dummy_source",
        order=1,
        strategy=strategy,
        label="DUMMY",
    )
    sources = [src_cfg]
    ctx_builder = RAGContextBuilder(sources=sources)

    # 2) Setup dummy chat client
    chat = DummyChatClient()

    # 3) Minimal TRF-like dict
    trf = {"meta": {"id": 123}, "data": "x"}

    # 4) Run pipeline
    answer = run_single_rag_analysis(
        trf=trf,
        query="why did it fail?",
        task_prompt="Explain what happened.",
        system_prompt="You are a test system.",
        sources=sources,
        context_builder=ctx_builder,
        chat_client=chat,
        max_trf_json_chars=None,
    )

    # 5) Assert we got dummy answer, and prompt was built
    assert answer.startswith("[DUMMY-ANSWER]")
    assert "TRF JSON" in chat.last_user_content
    assert "RAG CONTEXT" in chat.last_user_content
    assert "Explain what happened." in chat.last_user_content
