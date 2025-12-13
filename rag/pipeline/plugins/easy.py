# rag/pipeline/plugins/easy.py
from __future__ import annotations

from dataclasses import dataclass

from rag.config.schema import RAGConfig
from rag.pipeline.base import Pipeline, PipelinePlugin
from rag.pipeline.engine import RAGPipeline

from core.logging.logger import get_logger
from core.logging.tags import PIPELINE

logger = get_logger(__name__)


@dataclass
class EasyPipelinePlugin(PipelinePlugin):
    """
    Minimal convenience pipeline plugin.

    Must only delegate to RAGPipeline.from_config().
    """

    plugin_name: str = "easy"

    def build(self, cfg: RAGConfig) -> Pipeline:
        logger.info(f"{PIPELINE} Building Easy pipeline")
        return RAGPipeline.from_config(cfg)
