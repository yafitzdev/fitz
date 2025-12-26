# fitz_ai/ingest/enrichment/artifacts/plugins/data_model_reference.py
"""
Data model reference artifact generator.

Generates a reference of data models and their fields.
Helps answer "what fields does X have?" questions.
"""

from __future__ import annotations

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.base import ContentType

plugin_name = "data_model_reference"
plugin_type = "artifact"
description = "Data models with their fields"
supported_types = {ContentType.PYTHON}
requires_llm = False


class Generator:
    """
    Generates a data model reference: models with fields.

    This artifact helps answer "what fields does X have?" questions.
    """

    artifact_type = ArtifactType.DATA_MODEL_REFERENCE
    supported_types = supported_types

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
