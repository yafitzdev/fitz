# fitz_ai/ingest/enrichment/artifacts/plugins/navigation_index.py
"""
Navigation index artifact generator.

Generates a file → purpose mapping for quick navigation.
Helps answer "where is X?" questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.base import ContentType

plugin_name = "navigation_index"
plugin_type = "artifact"
description = "File to purpose mapping for quick navigation"
supported_types = {ContentType.PYTHON, ContentType.CODE, ContentType.DOCUMENT}
requires_llm = False


class Generator:
    """
    Generates a navigation index: file → purpose mapping.

    This artifact helps answer "where is X?" questions.
    """

    plugin_name = plugin_name
    artifact_type = ArtifactType.NAVIGATION_INDEX
    supported_types = supported_types

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate navigation index artifact."""
        lines = ["# Navigation Index", ""]
        lines.append("File → Purpose mapping for quick navigation.")
        lines.append("")

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
        if file["docstring"]:
            first_line = file["docstring"].split("\n")[0].strip()
            if len(first_line) > 10:
                return first_line[:100] + ("..." if len(first_line) > 100 else "")

        if file["exports"]:
            exports_str = ", ".join(file["exports"][:3])
            if len(file["exports"]) > 3:
                exports_str += f" (+{len(file['exports']) - 3} more)"
            return f"Defines: {exports_str}"

        return "(no description)"
