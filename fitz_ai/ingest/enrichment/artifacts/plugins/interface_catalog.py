# fitz_ai/ingest/enrichment/artifacts/plugins/interface_catalog.py
"""
Interface catalog artifact generator.

Generates a catalog of protocols and their implementations.
Helps answer "how do I implement X?" questions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.base import ContentType

plugin_name = "interface_catalog"
plugin_type = "artifact"
description = "Protocols and their implementations"
supported_types = {ContentType.PYTHON, ContentType.CODE}
requires_llm = False


class Generator:
    """
    Generates an interface catalog: protocols with implementations.

    This artifact helps answer "how do I implement X?" questions.
    """

    artifact_type = ArtifactType.INTERFACE_CATALOG
    supported_types = supported_types

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
