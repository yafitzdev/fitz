from __future__ import annotations

"""
fitz_rag

Public package interface for the Fitz-RAG toolkit.
Only stable, user-facing entry points are exported here.
"""

# -------------------------
# Config
# -------------------------
from .config.loader import get_config
from .config.schema import RAGConfig

# -------------------------
# Core Pipeline Engine
# -------------------------
from .pipeline.engine import RAGPipeline

# -------------------------
# Pipeline Plugins (User-selectable pipeline builders)
# -------------------------
from .pipeline.plugins.easy import EasyPipelinePlugin
from .pipeline.plugins.fast import FastPipelinePlugin
from .pipeline.plugins.standard import StandardPipelinePlugin
from .pipeline.plugins.debug import DebugPipelinePlugin

# -------------------------
# Context Builder
# -------------------------
from .context.pipeline import ContextPipeline

# -------------------------
# RGS
# -------------------------
from .generation.rgs import RGS, RGSConfig

__all__ = [
    "get_config",
    "RAGConfig",
    "RAGPipeline",

    # pipeline plugins
    "EasyPipelinePlugin",
    "FastPipelinePlugin",
    "StandardPipelinePlugin",
    "DebugPipelinePlugin",

    "ContextPipeline",
    "RGS",
    "RGSConfig",
]

__version__ = "0.1.0"
