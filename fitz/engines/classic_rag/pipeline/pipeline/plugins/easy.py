# pipeline/pipeline/plugins/easy.py
from __future__ import annotations

from dataclasses import dataclass

from fitz.logging.logger import get_logger
from fitz.logging.tags import PIPELINE
from fitz.engines.classic_rag.config.schema import RAGConfig
from fitz.engines.classic_rag.pipeline.pipeline.base import Pipeline, PipelinePlugin
from fitz.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline

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
