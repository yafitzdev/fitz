# fitz_ai/ingest/enrichment/__init__.py
"""
Enrichment module for enhancing document ingestion.

This module provides:
- EnrichmentPipeline: Unified entry point for all enrichment operations
- EnrichmentConfig: Configuration for enrichment behavior
- Chunk summaries: LLM-generated descriptions for better search
- Artifacts: Project-level insights (navigation index, interface catalog, etc.)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    EnrichmentPipeline                        │
    │            (single entry point for all enrichment)           │
    └─────────────────────────────────────────────────────────────┘
                               │
           ┌───────────────────┴───────────────────┐
           │                                       │
           ▼                                       ▼
    ┌──────────────┐                       ┌──────────────┐
    │   Summary    │                       │  Artifacts   │
    │  (chunks)    │                       │  (project)   │
    └──────────────┘                       └──────────────┘
           │                                       │
           ▼                                       ▼
    ┌──────────────┐                       ┌──────────────┐
    │   Context    │                       │   Plugins    │
    │   Builders   │                       │ (per type)   │
    └──────────────┘                       └──────────────┘

Usage:
    from fitz_ai.ingest.enrichment import EnrichmentPipeline, EnrichmentConfig

    # Create pipeline
    pipeline = EnrichmentPipeline.from_config(
        config=config.get("enrichment"),
        project_root=Path("./src"),
        chat_client=my_llm,
    )

    # Unified enrichment (preferred - the "box" interface)
    result = pipeline.enrich(chunks)
    enriched_chunks = result.chunks    # Chunks with summaries in metadata
    artifacts = result.artifacts        # Corpus-level artifacts

    # Or use individual methods if needed
    artifacts = pipeline.generate_artifacts()
    description = pipeline.summarize_chunk(content, file_path, content_hash)

Plugin System:
    - Artifact plugins: fitz_ai/ingest/enrichment/artifacts/plugins/
    - Context plugins: fitz_ai/ingest/enrichment/context/plugins/

    Each plugin is auto-discovered and must define:
    - plugin_name: str
    - plugin_type: str ("artifact" or "context")
    - supported_types or supported_extensions
"""

# Configuration
from fitz_ai.ingest.enrichment.config import (
    EnrichmentConfig,
    SummaryConfig,
    ArtifactConfig,
)

# Pipeline (main entry point)
from fitz_ai.ingest.enrichment.pipeline import (
    EnrichmentPipeline,
)

# Models
from fitz_ai.ingest.enrichment.models import (
    EnrichmentResult,
)

# Base types
from fitz_ai.ingest.enrichment.base import (
    ContentType,
    EnrichmentContext,
    CodeEnrichmentContext,
    DocumentEnrichmentContext,
    ContextBuilder,
    Enricher,
)

# Summary components
from fitz_ai.ingest.enrichment.summary import (
    ChunkSummarizer,
    SummaryCache,
)

# Artifact components
from fitz_ai.ingest.enrichment.artifacts import (
    Artifact,
    ArtifactType,
    ArtifactGenerator,
    ProjectAnalysis,
    ProjectAnalyzer,
)

# Registries
from fitz_ai.ingest.enrichment.artifacts.registry import (
    ArtifactRegistry,
    get_artifact_registry,
    get_artifact_plugin,
    list_artifact_plugins,
)
from fitz_ai.ingest.enrichment.context.registry import (
    ContextRegistry,
    get_context_registry,
    get_context_plugin,
    list_context_plugins,
)

__all__ = [
    # Configuration
    "EnrichmentConfig",
    "SummaryConfig",
    "ArtifactConfig",
    # Pipeline
    "EnrichmentPipeline",
    # Models
    "EnrichmentResult",
    # Base types
    "ContentType",
    "EnrichmentContext",
    "CodeEnrichmentContext",
    "DocumentEnrichmentContext",
    "ContextBuilder",
    "Enricher",
    # Summary
    "ChunkSummarizer",
    "SummaryCache",
    # Artifacts
    "Artifact",
    "ArtifactType",
    "ArtifactGenerator",
    "ProjectAnalysis",
    "ProjectAnalyzer",
    # Registries
    "ArtifactRegistry",
    "get_artifact_registry",
    "get_artifact_plugin",
    "list_artifact_plugins",
    "ContextRegistry",
    "get_context_registry",
    "get_context_plugin",
    "list_context_plugins",
]
