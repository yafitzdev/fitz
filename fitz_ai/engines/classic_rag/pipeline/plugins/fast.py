# fitz_ai/engines/classic_rag/pipeline/plugins/fast.py
from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.config.schema import ClassicRagConfig
from fitz_ai.engines.classic_rag.pipeline.base import Pipeline, PipelinePlugin
from fitz_ai.engines.classic_rag.pipeline.engine import RAGPipeline
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

logger = get_logger(__name__)


def _clone_cfg(cfg: ClassicRagConfig) -> ClassicRagConfig:
    return cfg.model_copy(deep=True)


@dataclass
class FastPipelinePlugin(PipelinePlugin):
    """
    Low-latency pipeline plugin.

    Allowed behavior:
    - mutate config (disable rerank)
    - delegate to RAGPipeline.from_config()
    """

    plugin_name: str = "fast"

    def build(self, cfg: ClassicRagConfig) -> Pipeline:
        logger.info(f"{PIPELINE} Building Fast pipeline (rerank disabled)")

        clone = _clone_cfg(cfg)
        clone.rerank.enabled = False
        return RAGPipeline.from_config(clone)
