# fitz_ai/ingest/enrichment/artifacts/plugins/dependency_summary.py
"""
Dependency summary artifact generator.

Generates a summary of module dependencies.
Helps answer "what uses X?" questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.base import ContentType

plugin_name = "dependency_summary"
plugin_type = "artifact"
description = "Module dependency graph"
supported_types = {ContentType.PYTHON, ContentType.CODE}
requires_llm = False


class Generator:
    """
    Generates a dependency summary: what depends on what.

    This artifact helps answer "what uses X?" questions.
    """

    plugin_name = plugin_name
    artifact_type = ArtifactType.DEPENDENCY_SUMMARY
    supported_types = supported_types

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate dependency summary artifact."""
        lines = ["# Dependency Summary", ""]
        lines.append("High-level view of module dependencies.")
        lines.append("")

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
