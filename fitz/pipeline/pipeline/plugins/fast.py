# pipeline/pipeline/plugins/fast.py
from __future__ import annotations

import copy
from dataclasses import dataclass

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE
from fitz.pipeline.config.schema import RAGConfig
from fitz.pipeline.pipeline.base import Pipeline, PipelinePlugin
from fitz.pipeline.pipeline.engine import RAGPipeline

logger = get_logger(__name__)


def _clone_cfg(cfg: RAGConfig) -> RAGConfig:
    if hasattr(cfg, "model_copy"):
        return cfg.model_copy(deep=True)  # pydantic v2
    if hasattr(cfg, "copy"):
        try:
            return cfg.copy(deep=True)  # pydantic v1
        except TypeError:
            return cfg.copy()
    return copy.deepcopy(cfg)


@dataclass
class FastPipelinePlugin(PipelinePlugin):
    """
    Low-latency pipeline plugin.

    Allowed behavior:
    - mutate config (disable rerank)
    - delegate to RAGPipeline.from_config()
    """

    plugin_name: str = "fast"

    def build(self, cfg: RAGConfig) -> Pipeline:
        logger.info(f"{PIPELINE} Building Fast pipeline (rerank disabled)")

        clone = _clone_cfg(cfg)
        clone.rerank.enabled = False
        return RAGPipeline.from_config(clone)
