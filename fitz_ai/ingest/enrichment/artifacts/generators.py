# fitz_ai/ingest/enrichment/artifacts/generators.py
"""
Artifact generators for different artifact types.

Each generator takes a ProjectAnalysis and produces an Artifact.
Some generators use LLM calls for summarization, others are purely structural.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ProjectAnalysis,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def complete(self, prompt: str) -> str: ...


class NavigationIndexGenerator:
    """
    Generates a navigation index: file → purpose mapping.

    This artifact helps answer "where is X?" questions.
    Uses LLM to summarize each file's purpose in one line.
    """

    artifact_type = ArtifactType.NAVIGATION_INDEX

    def __init__(self, chat_client: ChatClient | None = None):
        self._chat = chat_client

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate navigation index artifact."""
        lines = ["# Navigation Index", ""]
        lines.append("File → Purpose mapping for quick navigation.")
        lines.append("")

        # Group by directory
        by_dir: Dict[str, List[Dict[str, Any]]] = {}
        for file_info in analysis.files:
            dir_path = str(Path(file_info.relative_path).parent)
            if dir_path not in by_dir:
                by_dir[dir_path] = []
            by_dir[dir_path].append(
                {
                    "path": file_info.relative_path,
                    "name": Path(file_info.relative_path).name,
                    "docstring": file_info.docstring,
                    "exports": file_info.exports,
                }
            )

        for dir_path in sorted(by_dir.keys()):
            if dir_path == ".":
                lines.append("## Root")
            else:
                lines.append(f"## {dir_path}/")
            lines.append("")

            for file in sorted(by_dir[dir_path], key=lambda f: f["name"]):
                purpose = self._get_purpose(file)
                lines.append(f"- `{file['name']}` → {purpose}")

            lines.append("")

        content = "\n".join(lines)

        return Artifact(
            artifact_type=self.artifact_type,
            title="Navigation Index",
            content=content,
            metadata={"file_count": len(analysis.files)},
        )

    def _get_purpose(self, file: Dict[str, Any]) -> str:
        """Get purpose description for a file."""
        # Try to use docstring first line
        if file["docstring"]:
            first_line = file["docstring"].split("\n")[0].strip()
            if len(first_line) > 10:
                return first_line[:100] + ("..." if len(first_line) > 100 else "")

        # Fall back to exports
        if file["exports"]:
            exports_str = ", ".join(file["exports"][:3])
            if len(file["exports"]) > 3:
                exports_str += f" (+{len(file['exports']) - 3} more)"
            return f"Defines: {exports_str}"

        return "(no description)"


class InterfaceCatalogGenerator:
    """
    Generates an interface catalog: protocols with implementations.

    This artifact helps answer "how do I implement X?" questions.
    """

    artifact_type = ArtifactType.INTERFACE_CATALOG

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate interface catalog artifact."""
        lines = ["# Interface Catalog", ""]
        lines.append("Protocols and interfaces available for implementation.")
        lines.append("")

        if not analysis.protocols:
            lines.append("No protocols found in the codebase.")
        else:
            for proto in sorted(analysis.protocols, key=lambda p: p["name"]):
                lines.append(f"## {proto['name']}")
                lines.append(f"*Defined in: `{proto['file_path']}`*")
                lines.append("")

                if proto.get("docstring"):
                    lines.append(proto["docstring"].split("\n")[0])
                    lines.append("")

                if proto.get("methods"):
                    lines.append("**Methods:**")
                    for method in proto["methods"]:
                        lines.append(
                            f"- `{method['name']}{method.get('signature', '()')}`"
                        )
                    lines.append("")

                # Find implementations
                implementations = self._find_implementations(proto["name"], analysis)
                if implementations:
                    lines.append("**Implementations:**")
                    for impl in implementations:
                        lines.append(f"- `{impl['name']}` in `{impl['file_path']}`")
                    lines.append("")

        content = "\n".join(lines)

        return Artifact(
            artifact_type=self.artifact_type,
            title="Interface Catalog",
            content=content,
            metadata={"protocol_count": len(analysis.protocols)},
        )

    def _find_implementations(
        self, protocol_name: str, analysis: ProjectAnalysis
    ) -> List[Dict[str, Any]]:
        """Find classes that implement a protocol."""
        implementations = []
        for cls in analysis.classes:
            if protocol_name in cls.get("bases", []):
                implementations.append(cls)
        return implementations


class DataModelReferenceGenerator:
    """
    Generates a data model reference: models with fields.

    This artifact helps answer "what fields does X have?" questions.
    """

    artifact_type = ArtifactType.DATA_MODEL_REFERENCE

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate data model reference artifact."""
        lines = ["# Data Model Reference", ""]
        lines.append("Core data models and their fields.")
        lines.append("")

        if not analysis.models:
            lines.append("No Pydantic models found in the codebase.")
        else:
            for model in sorted(analysis.models, key=lambda m: m["name"]):
                lines.append(f"## {model['name']}")
                lines.append(f"*Defined in: `{model['file_path']}`*")
                lines.append("")

                if model.get("docstring"):
                    lines.append(model["docstring"].split("\n")[0])
                    lines.append("")

                if model.get("fields"):
                    lines.append("**Fields:**")
                    for field in model["fields"]:
                        lines.append(f"- `{field['name']}`: `{field['type']}`")
                    lines.append("")

        content = "\n".join(lines)

        return Artifact(
            artifact_type=self.artifact_type,
            title="Data Model Reference",
            content=content,
            metadata={"model_count": len(analysis.models)},
        )


class DependencySummaryGenerator:
    """
    Generates a dependency summary: what depends on what.

    This artifact helps answer "what uses X?" questions.
    """

    artifact_type = ArtifactType.DEPENDENCY_SUMMARY

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate dependency summary artifact."""
        lines = ["# Dependency Summary", ""]
        lines.append("High-level view of module dependencies.")
        lines.append("")

        # Compute package-level dependencies
        pkg_deps = self._compute_package_deps(analysis)

        if not pkg_deps:
            lines.append("No significant dependencies detected.")
        else:
            lines.append("## Package Dependencies")
            lines.append("")
            for pkg, deps in sorted(pkg_deps.items()):
                if deps:
                    deps_str = ", ".join(sorted(deps))
                    lines.append(f"- `{pkg}` → {deps_str}")
                else:
                    lines.append(f"- `{pkg}` → (no internal dependencies)")
            lines.append("")

        # Add import statistics
        lines.append("## Import Statistics")
        lines.append("")
        most_imported = self._find_most_imported(analysis)
        if most_imported:
            lines.append("Most imported modules:")
            for module, count in most_imported[:10]:
                lines.append(f"- `{module}`: {count} imports")
        lines.append("")

        content = "\n".join(lines)

        return Artifact(
            artifact_type=self.artifact_type,
            title="Dependency Summary",
            content=content,
            metadata={"package_count": len(pkg_deps)},
        )

    def _compute_package_deps(self, analysis: ProjectAnalysis) -> Dict[str, set]:
        """Compute package-level dependencies."""
        pkg_deps: Dict[str, set] = {}

        for file_info in analysis.files:
            parts = Path(file_info.relative_path).parts
            if len(parts) < 2:
                continue

            pkg = parts[0]
            if pkg not in pkg_deps:
                pkg_deps[pkg] = set()

            for imp in file_info.imports:
                imp_parts = imp.split(".")
                if imp_parts[0] in analysis.packages and imp_parts[0] != pkg:
                    pkg_deps[pkg].add(imp_parts[0])

        return pkg_deps

    def _find_most_imported(self, analysis: ProjectAnalysis) -> List[tuple]:
        """Find the most frequently imported modules."""
        import_counts: Dict[str, int] = {}

        for file_info in analysis.files:
            for imp in file_info.imports:
                import_counts[imp] = import_counts.get(imp, 0) + 1

        return sorted(import_counts.items(), key=lambda x: -x[1])


class ArchitectureNarrativeGenerator:
    """
    Generates an architecture narrative: high-level system overview.

    This artifact uses an LLM to generate a coherent description
    of how the system works based on the analyzed structure.
    """

    artifact_type = ArtifactType.ARCHITECTURE_NARRATIVE

    def __init__(self, chat_client: ChatClient):
        self._chat = chat_client

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate architecture narrative artifact."""
        # Build context for LLM
        context = self._build_context(analysis)

        prompt = f"""You are documenting a Python codebase for a search index. Based on the following analysis, write a clear architecture overview.

{context}

Write 3-5 paragraphs covering:
1. What this project does (purpose)
2. How it's organized (main packages and their roles)
3. Key abstractions (protocols, interfaces)
4. How data flows through the system
5. How to extend it (plugin points)

Be specific and technical. This will help developers understand the codebase quickly.
"""

        narrative = self._chat.complete(prompt)

        # Build structured content
        lines = ["# Architecture Overview", ""]
        lines.append(narrative)
        lines.append("")

        # Add quick reference
        lines.append("## Quick Reference")
        lines.append("")
        lines.append("**Packages:**")
        for pkg in sorted(analysis.packages):
            lines.append(f"- `{pkg}`")
        lines.append("")

        if analysis.protocols:
            lines.append("**Key Protocols:**")
            for proto in analysis.protocols[:5]:
                lines.append(f"- `{proto['name']}` ({proto['file_path']})")
            if len(analysis.protocols) > 5:
                lines.append(f"- ... and {len(analysis.protocols) - 5} more")
        lines.append("")

        content = "\n".join(lines)

        return Artifact(
            artifact_type=self.artifact_type,
            title="Architecture Overview",
            content=content,
            metadata={
                "package_count": len(analysis.packages),
                "protocol_count": len(analysis.protocols),
                "model_count": len(analysis.models),
            },
        )

    def _build_context(self, analysis: ProjectAnalysis) -> str:
        """Build context string for LLM."""
        lines = []

        lines.append("## Project Structure")
        lines.append(f"Packages: {', '.join(analysis.packages)}")
        lines.append(f"Files: {len(analysis.files)}")
        lines.append("")

        lines.append("## Protocols (Interfaces)")
        for proto in analysis.protocols[:10]:
            methods = [m["name"] for m in proto.get("methods", [])]
            lines.append(f"- {proto['name']}: {', '.join(methods[:3])}")
        lines.append("")

        lines.append("## Data Models")
        for model in analysis.models[:10]:
            fields = [f["name"] for f in model.get("fields", [])]
            lines.append(f"- {model['name']}: {', '.join(fields[:5])}")
        lines.append("")

        lines.append("## Key Files (by docstring)")
        for file in analysis.files[:15]:
            if file.docstring:
                first_line = file.docstring.split("\n")[0][:80]
                lines.append(f"- {file.relative_path}: {first_line}")
        lines.append("")

        return "\n".join(lines)


__all__ = [
    "NavigationIndexGenerator",
    "InterfaceCatalogGenerator",
    "DataModelReferenceGenerator",
    "DependencySummaryGenerator",
    "ArchitectureNarrativeGenerator",
]
