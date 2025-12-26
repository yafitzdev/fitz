# fitz_ai/ingest/enrichment/artifacts/orchestrator.py
"""
Orchestrator for artifact generation.

Coordinates:
1. Project analysis
2. Running all artifact generators
3. Returning artifacts ready for ingestion
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from fitz_ai.ingest.enrichment.artifacts.analyzer import ProjectAnalyzer
from fitz_ai.ingest.enrichment.artifacts.base import Artifact, ProjectAnalysis
from fitz_ai.ingest.enrichment.artifacts.generators import (
    ArchitectureNarrativeGenerator,
    DataModelReferenceGenerator,
    DependencySummaryGenerator,
    InterfaceCatalogGenerator,
    NavigationIndexGenerator,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def complete(self, prompt: str) -> str: ...


class ArtifactOrchestrator:
    """
    Orchestrates artifact generation for a project.

    Usage:
        orchestrator = ArtifactOrchestrator(
            project_root=Path("/path/to/project"),
            chat_client=llm_client,
        )
        artifacts = orchestrator.generate_all()

        # Ingest artifacts
        for artifact in artifacts:
            # Store in vector DB
            pass
    """

    def __init__(
        self,
        project_root: Path,
        chat_client: ChatClient | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            project_root: Root directory of the project to analyze
            chat_client: LLM client for generators that need it
        """
        self._root = Path(project_root).resolve()
        self._chat = chat_client
        self._analysis: ProjectAnalysis | None = None

    def analyze(self) -> ProjectAnalysis:
        """Analyze the project (cached)."""
        if self._analysis is None:
            analyzer = ProjectAnalyzer(self._root)
            self._analysis = analyzer.analyze()
        return self._analysis

    def generate_all(self) -> List[Artifact]:
        """
        Generate all artifacts for the project.

        Returns:
            List of Artifact instances ready for ingestion
        """
        analysis = self.analyze()
        artifacts: List[Artifact] = []

        # Navigation Index (no LLM required)
        try:
            nav_gen = NavigationIndexGenerator()
            artifacts.append(nav_gen.generate(analysis))
            logger.info("Generated: Navigation Index")
        except Exception as e:
            logger.error(f"Failed to generate Navigation Index: {e}")

        # Interface Catalog (no LLM required)
        try:
            iface_gen = InterfaceCatalogGenerator()
            artifacts.append(iface_gen.generate(analysis))
            logger.info("Generated: Interface Catalog")
        except Exception as e:
            logger.error(f"Failed to generate Interface Catalog: {e}")

        # Data Model Reference (no LLM required)
        try:
            model_gen = DataModelReferenceGenerator()
            artifacts.append(model_gen.generate(analysis))
            logger.info("Generated: Data Model Reference")
        except Exception as e:
            logger.error(f"Failed to generate Data Model Reference: {e}")

        # Dependency Summary (no LLM required)
        try:
            dep_gen = DependencySummaryGenerator()
            artifacts.append(dep_gen.generate(analysis))
            logger.info("Generated: Dependency Summary")
        except Exception as e:
            logger.error(f"Failed to generate Dependency Summary: {e}")

        # Architecture Narrative (requires LLM)
        if self._chat:
            try:
                arch_gen = ArchitectureNarrativeGenerator(self._chat)
                artifacts.append(arch_gen.generate(analysis))
                logger.info("Generated: Architecture Narrative")
            except Exception as e:
                logger.error(f"Failed to generate Architecture Narrative: {e}")
        else:
            logger.warning("Skipping Architecture Narrative (no LLM client provided)")

        logger.info(f"Generated {len(artifacts)} artifacts")
        return artifacts

    def generate_structural(self) -> List[Artifact]:
        """
        Generate only structural artifacts (no LLM required).

        Useful for quick generation without API costs.
        """
        analysis = self.analyze()
        artifacts: List[Artifact] = []

        generators = [
            NavigationIndexGenerator(),
            InterfaceCatalogGenerator(),
            DataModelReferenceGenerator(),
            DependencySummaryGenerator(),
        ]

        for gen in generators:
            try:
                artifacts.append(gen.generate(analysis))
            except Exception as e:
                logger.error(f"Failed to generate {gen.artifact_type}: {e}")

        return artifacts


__all__ = ["ArtifactOrchestrator"]
