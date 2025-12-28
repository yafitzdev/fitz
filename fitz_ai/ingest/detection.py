# fitz_ai/ingest/detection.py
"""
Content type detection for ingestion.

Detects whether a directory contains a codebase or document corpus
to automatically configure the appropriate enrichment strategy:

- Codebase: Run structural analysis (codebase_analysis/artifacts)
- Documents: Run hierarchical summarization (hierarchy)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Project marker files that indicate a codebase
CODEBASE_MARKERS = {
    # Python
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "requirements.txt": "python",
    "Pipfile": "python",
    # Node/JavaScript
    "package.json": "node",
    "package-lock.json": "node",
    "yarn.lock": "node",
    "pnpm-lock.yaml": "node",
    # Rust
    "Cargo.toml": "rust",
    "Cargo.lock": "rust",
    # Go
    "go.mod": "go",
    "go.sum": "go",
    # .NET
    "*.sln": "dotnet",
    "*.csproj": "dotnet",
    "*.fsproj": "dotnet",
    # Java/JVM
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    # Ruby
    "Gemfile": "ruby",
    "Gemfile.lock": "ruby",
    # PHP
    "composer.json": "php",
    "composer.lock": "php",
}

ContentType = Literal["codebase", "documents"]


@dataclass
class DetectionResult:
    """Result of content type detection."""

    content_type: ContentType
    reason: str
    project_type: str | None = None  # e.g., "python", "node", "rust"


def detect_content_type(source_path: Path) -> DetectionResult:
    """
    Detect whether a directory contains a codebase or document corpus.

    Args:
        source_path: Path to the directory to analyze

    Returns:
        DetectionResult with content_type, reason, and optional project_type
    """
    source = Path(source_path).resolve()

    if not source.exists():
        return DetectionResult(
            content_type="documents",
            reason="Path does not exist, defaulting to documents",
        )

    if source.is_file():
        # Single file - treat as documents
        return DetectionResult(
            content_type="documents",
            reason="Single file ingestion",
        )

    # Check for project marker files
    for marker, project_type in CODEBASE_MARKERS.items():
        if "*" in marker:
            # Glob pattern (e.g., "*.sln")
            if list(source.glob(marker)):
                return DetectionResult(
                    content_type="codebase",
                    reason=f"Found {marker} (indicates {project_type} project)",
                    project_type=project_type,
                )
        else:
            # Exact file name
            if (source / marker).exists():
                return DetectionResult(
                    content_type="codebase",
                    reason=f"Found {marker} (indicates {project_type} project)",
                    project_type=project_type,
                )

    # No markers found - default to documents
    return DetectionResult(
        content_type="documents",
        reason="No codebase markers found, treating as document corpus",
    )


def is_codebase(source_path: Path) -> bool:
    """Quick check if a path is a codebase."""
    return detect_content_type(source_path).content_type == "codebase"


def is_documents(source_path: Path) -> bool:
    """Quick check if a path is a document corpus."""
    return detect_content_type(source_path).content_type == "documents"


__all__ = [
    "ContentType",
    "DetectionResult",
    "detect_content_type",
    "is_codebase",
    "is_documents",
    "CODEBASE_MARKERS",
]
