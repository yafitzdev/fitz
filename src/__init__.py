"""
fitz_rag – Retrieval-Augmented Generation Toolkit
Public API surface for v0.1.0
"""

# Core pipeline
from fitz_rag.pipeline.engine import (
    RAGPipeline,
    create_pipeline,
    create_pipeline_from_yaml,
)

# Preset pipelines
from fitz_rag.pipeline.easy import EasyRAG
from fitz_rag.pipeline.fast import FastRAG
from fitz_rag.pipeline.standard import StandardRAG
from fitz_rag.pipeline.debug import DebugRAG

# Retrieval-Guided Synthesis
from fitz_rag.generation.rgs import RGS, RGSConfig

# Core data models
from fitz_rag.models.chunk import Chunk
from fitz_rag.models.document import Document

# Unified configuration (Pydantic)
from fitz_rag.config.schema import (
    RAGConfig,
    LLMConfig,
    EmbeddingConfig,
    RerankConfig,
    RetrieverConfig,
    RGSSettings,
    LoggingConfig,
    IngestionConfig,
)

__all__ = [
    # Pipelines
    "RAGPipeline",
    "create_pipeline",
    "create_pipeline_from_yaml",
    "EasyRAG",
    "FastRAG",
    "StandardRAG",
    "DebugRAG",
    # RGS
    "RGS",
    "RGSConfig",
    # Models
    "Chunk",
    "Document",
    # Config
    "RAGConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "RerankConfig",
    "RetrieverConfig",
    "RGSSettings",
    "LoggingConfig",
    "IngestionConfig",
]
