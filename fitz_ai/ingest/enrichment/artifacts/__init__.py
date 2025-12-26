# fitz_ai/ingest/enrichment/artifacts/__init__.py
"""
Artifact generation for improved code retrieval.

Artifacts are high-level summaries of a codebase that provide context
for understanding code structure and relationships. They are:
1. Generated once per ingest (or when code changes significantly)
2. Stored in the vector DB with special metadata
3. Always retrieved with every query (score=1.0)

Artifact plugins are auto-discovered from the plugins/ directory.
Each plugin must define:
    - plugin_name: str
    - plugin_type: str = "artifact"
    - supported_types: set[ContentType]
    - requires_llm: bool
    - Generator class

Available artifact types (via plugins):
- navigation_index: File → purpose mapping
- interface_catalog: Protocols/interfaces with implementations
- data_model_reference: Core models with fields
- dependency_summary: What depends on what
- architecture_narrative: High-level system overview (requires LLM)

Usage:
    from fitz_ai.ingest.enrichment import EnrichmentPipeline

    pipeline = EnrichmentPipeline.from_config(
        config=config.get("enrichment"),
        project_root=Path("/path/to/project"),
        chat_client=llm_client,
    )
    artifacts = pipeline.generate_artifacts()
"""

from fitz_ai.ingest.enrichment.artifacts.analyzer import ProjectAnalyzer
from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactGenerator,
    ArtifactType,
    FileInfo,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.artifacts.registry import (
    ArtifactPluginInfo,
    ArtifactRegistry,
    get_artifact_plugin,
    get_artifact_registry,
    list_artifact_plugins,
)

__all__ = [
    # Base types
    "Artifact",
    "ArtifactType",
    "ArtifactGenerator",
    "FileInfo",
    "ProjectAnalysis",
    # Analyzer
    "ProjectAnalyzer",
    # Registry
    "ArtifactRegistry",
    "ArtifactPluginInfo",
    "get_artifact_registry",
    "get_artifact_plugin",
    "list_artifact_plugins",
]
