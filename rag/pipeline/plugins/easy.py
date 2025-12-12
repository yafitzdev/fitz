from __future__ import annotations

from dataclasses import dataclass

from rag.config.schema import RAGConfig
from rag.pipeline.engine import RAGPipeline
from rag.pipeline.base import PipelinePlugin

from core.logging import get_logger
from core.logging_tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class EasyPipelinePlugin(PipelinePlugin):
    """
    Minimal convenience pipeline plugin.

    Currently equivalent to the standard pipeline, but exists as a dedicated
    plugin so future "easy mode" behavior can be added without touching the
    core engine.
    """

    def build(self, cfg: RAGConfig) -> RAGPipeline:
        logger.info(f"{PIPELINE} Building Easy pipeline")
        return RAGPipeline.from_config(cfg)
