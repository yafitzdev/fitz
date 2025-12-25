# fitz_ai/ingest/enrichment/artifacts/base.py
"""
Base types for artifact generation.

Artifacts are high-level summaries that provide context for code retrieval.
Each artifact type serves a specific purpose in helping users find relevant code.

Artifact plugins are auto-discovered from the plugins/ directory. Each plugin
must define:
    - plugin_name: str - Unique identifier for the artifact
    - plugin_type: str - Always "artifact"
    - supported_types: set[ContentType] - Content types this artifact applies to
    - Generator class implementing ArtifactGenerator protocol
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, Set, runtime_checkable

from fitz_ai.ingest.enrichment.base import ContentType

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk


class ArtifactType(str, Enum):
    """Types of artifacts that can be generated."""

    ARCHITECTURE_NARRATIVE = "architecture_narrative"
    NAVIGATION_INDEX = "navigation_index"
    INTERFACE_CATALOG = "interface_catalog"
    DATA_MODEL_REFERENCE = "data_model_reference"
    DEPENDENCY_SUMMARY = "dependency_summary"


@dataclass
class Artifact:
    """
    A generated artifact ready for ingestion.

    Attributes:
        artifact_type: The type of artifact
        title: Human-readable title
        content: The artifact content (will be embedded)
        metadata: Additional metadata for filtering/display
    """

    artifact_type: ArtifactType
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_artifact(self) -> bool:
        """Always True for artifacts."""
        return True

    def to_payload(self) -> Dict[str, Any]:
        """Convert to vector DB payload format."""
        return {
            "content": self.content,
            "is_artifact": True,
            "artifact_type": self.artifact_type.value,
            "title": self.title,
            "doc_id": f"artifact:{self.artifact_type.value}",
            "chunk_index": 0,
            **self.metadata,
        }

    def to_chunk(self) -> "Chunk":
        """
        Convert artifact to a Chunk for vector DB storage.

        This enables unified storage flow where artifacts are stored
        alongside regular chunks in the vector database.

        Returns:
            Chunk instance with artifact metadata
        """
        from fitz_ai.engines.classic_rag.models.chunk import Chunk

        artifact_id = hashlib.sha256(
            f"{self.artifact_type.value}:{self.title}".encode()
        ).hexdigest()[:16]

        return Chunk(
            id=f"artifact:{artifact_id}",
            doc_id=f"artifact:{self.artifact_type.value}",
            content=self.content,
            chunk_index=0,
            metadata={
                "is_artifact": True,
                "artifact_type": self.artifact_type.value,
                "title": self.title,
                **self.metadata,
            },
        )


@runtime_checkable
class ArtifactGenerator(Protocol):
    """
    Protocol for artifact generators.

    Each generator is responsible for:
    1. Analyzing the codebase
    2. Generating a specific type of artifact
    3. Returning an Artifact instance

    Generators can use AST parsing, LLM calls, or both.

    Plugin attributes (module-level):
        plugin_name: str - Unique identifier (e.g., "navigation_index")
        plugin_type: str - Always "artifact"
        supported_types: set[ContentType] - Content types this applies to
        requires_llm: bool - Whether this generator needs an LLM client
    """

    artifact_type: ArtifactType
    supported_types: Set[ContentType]

    def generate(self, analysis: "ProjectAnalysis") -> Artifact:
        """
        Generate an artifact from project analysis.

        Args:
            analysis: Pre-computed project analysis data

        Returns:
            Generated Artifact instance
        """
        ...


@dataclass
class FileInfo:
    """Information about a single file for artifact generation."""

    path: str
    relative_path: str
    extension: str
    size_bytes: int
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    docstring: str | None = None


@dataclass
class ProjectAnalysis:
    """
    Analyzed project information for artifact generation.

    This is the common data structure that all generators can use.
    It's generated once and shared across generators.
    """

    root: Path
    files: List[FileInfo] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)

    # Extracted structures
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    protocols: List[Dict[str, Any]] = field(default_factory=list)
    models: List[Dict[str, Any]] = field(default_factory=list)

    # Import graph
    import_graph: Dict[str, List[str]] = field(default_factory=dict)


__all__ = [
    "ArtifactType",
    "Artifact",
    "ArtifactGenerator",
    "FileInfo",
    "ProjectAnalysis",
]
