# pipeline/pipeline/plugins/easy.py
from __future__ import annotations

from dataclasses import dataclass

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE
from fitz.pipeline.config.schema import RAGConfig
from fitz.pipeline.pipeline.base import Pipeline, PipelinePlugin
from fitz.pipeline.pipeline.engine import RAGPipeline

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
