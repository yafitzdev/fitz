from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from rag.config.schema import RAGConfig
from rag.pipeline.engine import RAGPipeline
from rag.pipeline.base import PipelinePlugin

from core.logging import get_logger
from core.logging_tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class DebugPipelinePlugin(PipelinePlugin):
    """
    Debug pipeline plugin.

    Wraps a RAGPipeline in a DebugRunner that exposes an `explain(query)`
    method for introspection while still supporting `.run(query)`.
    """

    def build(self, cfg: RAGConfig) -> "DebugRunner":
        logger.info(f"{PIPELINE} Building Debug pipeline")
        pipeline = RAGPipeline.from_config(cfg)
        return DebugRunner(pipeline)


class DebugRunner:
    """
    Wrapper around RAGPipeline providing step-by-step introspection.
    """

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(self, query: str):
        return self.pipeline.run(query)

    def explain(self, query: str) -> Dict[str, Any]:
        logger.info(f"{PIPELINE} DebugRunner.explain for query='{query}'")

        # 1) Retrieve
        chunks = self.pipeline.retriever.retrieve(query)

        # 2) Build RGS prompt
        prompt = self.pipeline.rgs.build_prompt(query, chunks)
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3) LLM call
        raw = self.pipeline.llm.chat(messages)

        return {
            "query": query,
            "chunks": chunks,
            "prompt": prompt,
            "messages": messages,
            "answer_raw": raw,
        }
