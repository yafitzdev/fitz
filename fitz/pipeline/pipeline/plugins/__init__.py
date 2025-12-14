# pipeline/pipeline/plugins/__init__.py
from __future__ import annotations

from .debug import DebugPipelinePlugin
from .easy import EasyPipelinePlugin
from .fast import FastPipelinePlugin
from .standard import StandardPipelinePlugin

__all__ = [
    "DebugPipelinePlugin",
    "EasyPipelinePlugin",
    "FastPipelinePlugin",
    "StandardPipelinePlugin",
]
