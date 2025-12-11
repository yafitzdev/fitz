# fitz_rag/pipeline/debug.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config
from fitz_rag.pipeline.engine import RAGPipeline

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class DebugRAG:
    """
    Debug pipeline wrapper providing introspection.

    This is a thin helper around RAGPipeline with tools to inspect:
        - retrieved chunks
        - RGS prompt
        - model answer

    It has NO legacy parameters anymore.
    """

    config: Optional[RAGConfig] = None
    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        logger.info(f"{PIPELINE} Initializing DebugRAG")

        if self.config is None:
            raw = load_config()
            self.config = RAGConfig.from_dict(raw)

        self.pipeline = RAGPipeline.from_config(self.config)

    # ---------------------------------------------------------
    # Debug helper
    # ---------------------------------------------------------
    def explain(self, query: str) -> Dict[str, Any]:
        """
        Returns:
            {
                "query": query,
                "chunks": <retrieved chunks>,
                "prompt": <RGSPrompt>,
                "answer_raw": <LLM output text>,
            }
        """
        logger.info(f"{PIPELINE} DebugRAG.explain called for query='{query}'")

        # 1) Retrieve
        chunks = self.pipeline.retriever.retrieve(query)

        # 2) Build prompt
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
            "answer_raw": raw,
        }
