# pipeline/pipeline/plugins/debug.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE
from fitz.pipeline.config.schema import RAGConfig
from fitz.pipeline.pipeline.base import Pipeline, PipelinePlugin
from fitz.pipeline.pipeline.engine import RAGPipeline

logger = get_logger(__name__)


@dataclass
class DebugPipelinePlugin(PipelinePlugin):
    """
    Debug pipeline plugin.

    Wraps a RAGPipeline in a DebugRunner that exposes an `explain(query)` method.
    DebugRunner MUST use the same wiring (RAGPipeline.from_config()).
    """

    plugin_name: str = "debug"

    def build(self, cfg: RAGConfig) -> Pipeline:
        logger.info(f"{PIPELINE} Building Debug pipeline")
        return DebugRunner(RAGPipeline.from_config(cfg))


class DebugRunner:
    """
    Wrapper around RAGPipeline providing introspection.
    """

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def run(self, query: str):
        return self._pipeline.run(query)

    def explain(self, query: str) -> Dict[str, Any]:
        logger.info(f"{PIPELINE} DebugRunner.explain for query='{query}'")
        chunks = self._pipeline.retriever.retrieve(query)
        chunks = self._pipeline.context.process(chunks)

        prompt = self._pipeline.rgs.build_prompt(query, chunks)
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        raw = self._pipeline.llm.chat(messages)

        return {
            "query": query,
            "chunks": chunks,
            "prompt": prompt,
            "messages": messages,
            "answer_raw": raw,
        }
