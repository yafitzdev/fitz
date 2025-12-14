# pipeline/pipeline/base.py
from __future__ import annotations

from typing import Protocol

from fitz.pipeline.config.schema import RAGConfig


class Pipeline(Protocol):
    def run(self, query: str): ...


class PipelinePlugin(Protocol):
    """
    Contract:
    - Plugins do NOT implement wiring.
    - Plugins may only adjust config, then delegate to RAGPipeline.from_config(cfg).
    """

    plugin_name: str

    def build(self, cfg: RAGConfig) -> Pipeline: ...
