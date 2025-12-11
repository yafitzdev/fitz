from __future__ import annotations

"""
fitz_rag

Public package interface for the Fitz-RAG toolkit.
We expose only the modern, stable entry points here.
"""

from .config.loader import get_config  # convenience
from .config.schema import RAGConfig

from .pipeline.engine import RAGPipeline
from .pipeline.easy import EasyRAG
from .pipeline.fast import FastRAG
from .pipeline.standard import StandardRAG
from .pipeline.debug import DebugRAG

from .context.pipeline import ContextPipeline
from .generation.rgs import RGS, RGSConfig

__all__ = [
    "get_config",
    "RAGConfig",
    "RAGPipeline",
    "EasyRAG",
    "FastRAG",
    "StandardRAG",
    "DebugRAG",
    "ContextPipeline",
    "RGS",
    "RGSConfig",
]

# Optional but nice to have for CLI / introspection
__version__ = "0.1.0"
