from __future__ import annotations

from typing import Dict

from rag.pipeline.base import PipelinePlugin
from rag.pipeline.plugins.debug import DebugPipelinePlugin
from rag.pipeline.plugins.easy import EasyPipelinePlugin
from rag.pipeline.plugins.fast import FastPipelinePlugin
from rag.pipeline.plugins.standard import StandardPipelinePlugin


_REGISTRY: Dict[str, PipelinePlugin] = {
    "debug": DebugPipelinePlugin(),
    "easy": EasyPipelinePlugin(),
    "fast": FastPipelinePlugin(),
    "standard": StandardPipelinePlugin(),
}


def get_pipeline_plugin(name: str) -> PipelinePlugin:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline plugin: {name!r}") from exc


def available_pipeline_plugins() -> list[str]:
    return sorted(_REGISTRY.keys())
