# fitz_ai/ingest/enrichment/artifacts/plugins/architecture_narrative.py
"""
Architecture narrative artifact generator.

Generates a high-level system overview using LLM.
Helps understand how the system works at a high level.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fitz_ai.ingest.enrichment.artifacts.base import (
    Artifact,
    ArtifactType,
    ProjectAnalysis,
)
from fitz_ai.ingest.enrichment.base import ContentType

plugin_name = "architecture_narrative"
plugin_type = "artifact"
description = "High-level system overview (requires LLM, adds ~5s per 20 chunks)"
supported_types = {ContentType.PYTHON, ContentType.CODE}
requires_llm = True


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class Generator:
    """
    Generates an architecture narrative: high-level system overview.

    This artifact uses an LLM to generate a coherent description
    of how the system works based on the analyzed structure.
    """

    artifact_type = ArtifactType.ARCHITECTURE_NARRATIVE
    supported_types = supported_types

    def __init__(self, chat_client: ChatClient):
        self._chat = chat_client

    def generate(self, analysis: ProjectAnalysis) -> Artifact:
        """Generate architecture narrative artifact."""
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

        messages = [{"role": "user", "content": prompt}]
        narrative = self._chat.chat(messages)

        lines = ["# Architecture Overview", ""]
        lines.append(narrative)
        lines.append("")

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
