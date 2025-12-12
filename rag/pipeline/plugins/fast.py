from __future__ import annotations

from dataclasses import dataclass

from rag.config.schema import RAGConfig, RerankConfig
from rag.pipeline.engine import RAGPipeline
from rag.pipeline.base import PipelinePlugin

from core.logging import get_logger
from core.logging_tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class FastPipelinePlugin(PipelinePlugin):
    """
    Low-latency pipeline plugin:

    - disables reranking
    - otherwise uses the same configuration as the standard pipeline
    """

    def build(self, cfg: RAGConfig) -> RAGPipeline:
        logger.info(f"{PIPELINE} Building Fast pipeline (rerank disabled)")

        clone = cfg.copy()
        clone.rerank = RerankConfig(
            provider=cfg.rerank.provider,
            model=cfg.rerank.model,
            api_key=cfg.rerank.api_key,
            enabled=False,
        )

        return RAGPipeline.from_config(clone)
