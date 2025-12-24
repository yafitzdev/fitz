# fitz_ai/ingest/enrichment/artifacts/__init__.py
"""
Artifact generation for improved code retrieval.

Artifacts are high-level summaries of a codebase that provide context
for understanding code structure and relationships. They are:
1. Generated once per ingest (or when code changes significantly)
2. Stored in the vector DB with special metadata
3. Always retrieved with every query (score=1.0)

Artifact Types:
- ArchitectureNarrative: High-level system overview
- NavigationIndex: File → purpose mapping
- InterfaceCatalog: Protocols/interfaces with implementations
- DataModelReference: Core models with fields
- DependencySummary: What depends on what

Usage:
    generator = ArtifactOrchestrator(
        project_root=Path("/path/to/project"),
        chat_client=llm_client,
    )
    artifacts = generator.generate_all()

    # Ingest artifacts
    for artifact in artifacts:
        ingest_artifact(artifact, collection, vector_db)
"""

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ArtifactGenerator,
    FileInfo,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.artifacts.analyzer import ProjectAnalyzer
from fitz_ai.ingest.enrichment.artifacts.generators import (
    ArchitectureNarrativeGenerator,
    DataModelReferenceGenerator,
    DependencySummaryGenerator,
    InterfaceCatalogGenerator,
    NavigationIndexGenerator,
)
from fitz_ai.ingest.enrichment.artifacts.orchestrator import ArtifactOrchestrator

__all__ = [
    # Base types
    "Artifact",
    "ArtifactType",
    "ArtifactGenerator",
    "FileInfo",
    "ProjectAnalysis",
    # Analyzer
    "ProjectAnalyzer",
    # Generators
    "ArchitectureNarrativeGenerator",
    "DataModelReferenceGenerator",
    "DependencySummaryGenerator",
    "InterfaceCatalogGenerator",
    "NavigationIndexGenerator",
    # Orchestrator
    "ArtifactOrchestrator",
]
