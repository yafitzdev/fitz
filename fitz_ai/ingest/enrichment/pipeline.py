# fitz_ai/ingest/enrichment/pipeline.py
"""
Unified enrichment pipeline.

The EnrichmentPipeline is the single entry point for all enrichment operations.
It coordinates:
1. Chunk-level summaries (universal)
2. Project-level artifacts (type-specific plugins)

Usage:
    from fitz_ai.ingest.enrichment import EnrichmentPipeline, EnrichmentConfig

    pipeline = EnrichmentPipeline.from_config(
        config=EnrichmentConfig.from_dict(config_dict.get("enrichment", {})),
        project_root=Path("/path/to/project"),
        chat_client=my_llm_client,
    )

    # Generate artifacts
    artifacts = pipeline.generate_artifacts()

    # Summarize a chunk
    description = pipeline.summarize_chunk(
        content="def hello(): ...",
        file_path="/path/to/file.py",
        content_hash="abc123",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Protocol, Set, runtime_checkable

from fitz_ai.ingest.enrichment.config import EnrichmentConfig
from fitz_ai.ingest.enrichment.base import ContentType
from fitz_ai.ingest.enrichment.summary.cache import SummaryCache
from fitz_ai.ingest.enrichment.summary.summarizer import ChunkSummarizer
from fitz_ai.ingest.enrichment.artifacts.base import Artifact, ProjectAnalysis
from fitz_ai.ingest.enrichment.artifacts.analyzer import ProjectAnalyzer
from fitz_ai.ingest.enrichment.artifacts.registry import (
    ArtifactRegistry,
    ArtifactPluginInfo,
)
from fitz_ai.core.paths import FitzPaths

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class EnrichmentPipeline:
    """
    Unified enrichment pipeline.

    Coordinates chunk-level summaries and project-level artifacts.

    Attributes:
        config: Enrichment configuration
        project_root: Root directory of the project
        summarizer: Chunk summarizer (if summaries enabled)
        artifact_registry: Registry of artifact plugins
    """

    def __init__(
        self,
        *,
        config: EnrichmentConfig,
        project_root: Path,
        chat_client: ChatClient | None = None,
        cache_path: Path | None = None,
        enricher_id: str | None = None,
    ):
        self.config = config
        self.project_root = Path(project_root).resolve()
        self._chat_client = chat_client
        self._cache_path = cache_path or FitzPaths.workspace() / "enrichment_cache.json"
        self._enricher_id = enricher_id or "default"

        self._summarizer: ChunkSummarizer | None = None
        self._artifact_registry = ArtifactRegistry.get_instance()
        self._project_analysis: ProjectAnalysis | None = None

        self._init_summarizer()

    def _init_summarizer(self) -> None:
        """Initialize the chunk summarizer if enabled."""
        if not self.config.enabled or not self.config.summary.enabled:
            return

        if self._chat_client is None:
            logger.warning("Summaries enabled but no chat client provided")
            return

        cache = SummaryCache(self._cache_path)
        self._summarizer = ChunkSummarizer(
            chat_client=self._chat_client,
            cache=cache,
            enricher_id=self._enricher_id,
        )

    @classmethod
    def from_config(
        cls,
        config: EnrichmentConfig | Dict[str, Any] | None,
        project_root: Path,
        chat_client: ChatClient | None = None,
        cache_path: Path | None = None,
        enricher_id: str | None = None,
    ) -> "EnrichmentPipeline":
        """
        Create pipeline from config.

        Args:
            config: EnrichmentConfig, dict, or None (uses defaults)
            project_root: Root directory of the project
            chat_client: LLM client for summaries and LLM-based artifacts
            cache_path: Path to cache file
            enricher_id: Unique identifier for cache invalidation
        """
        if config is None:
            config = EnrichmentConfig()
        elif isinstance(config, dict):
            config = EnrichmentConfig.from_dict(config)

        return cls(
            config=config,
            project_root=project_root,
            chat_client=chat_client,
            cache_path=cache_path,
            enricher_id=enricher_id,
        )

    @property
    def is_enabled(self) -> bool:
        """Check if enrichment is enabled."""
        return self.config.enabled

    @property
    def summaries_enabled(self) -> bool:
        """Check if chunk summaries are enabled."""
        return self.config.enabled and self.config.summary.enabled and self._summarizer is not None

    @property
    def artifacts_enabled(self) -> bool:
        """Check if artifacts are enabled."""
        return self.config.enabled

    def summarize_chunk(
        self,
        content: str,
        file_path: str,
        content_hash: str,
    ) -> str | None:
        """
        Generate a summary for a chunk.

        Args:
            content: The chunk content
            file_path: Path to the source file
            content_hash: Hash of the content (for caching)

        Returns:
            Summary string, or None if summaries are disabled
        """
        if not self.summaries_enabled:
            return None

        return self._summarizer.summarize(content, file_path, content_hash)

    def analyze_project(self) -> ProjectAnalysis:
        """
        Analyze the project (cached).

        Returns:
            ProjectAnalysis with extracted structure information
        """
        if self._project_analysis is None:
            analyzer = ProjectAnalyzer(self.project_root)
            self._project_analysis = analyzer.analyze()
        return self._project_analysis

    def get_applicable_artifact_plugins(self) -> List[ArtifactPluginInfo]:
        """
        Get artifact plugins applicable to the project.

        Filters based on:
        1. Content types present in the project
        2. Config enabled/disabled lists
        3. LLM availability for LLM-requiring plugins
        """
        all_plugins = self._artifact_registry.get_all_plugins()

        # Filter by content types in project
        # For now, assume code projects (most common case)
        # TODO: Detect actual content types from project
        project_types = {ContentType.PYTHON, ContentType.CODE}

        applicable = []
        for plugin in all_plugins:
            # Skip if no supported types match
            if not plugin.supported_types & project_types:
                continue

            # Skip if requires LLM but none available
            if plugin.requires_llm and self._chat_client is None:
                logger.debug(f"Skipping {plugin.name}: requires LLM")
                continue

            # Check config filters
            if self.config.artifacts.enabled:
                # Explicit list - only include if in list
                if plugin.name not in self.config.artifacts.enabled:
                    continue
            elif plugin.name in self.config.artifacts.disabled:
                # Auto mode - skip if in disabled list
                continue

            applicable.append(plugin)

        return applicable

    def generate_artifacts(self) -> List[Artifact]:
        """
        Generate all applicable artifacts.

        Returns:
            List of generated Artifact instances
        """
        if not self.artifacts_enabled:
            return []

        analysis = self.analyze_project()
        plugins = self.get_applicable_artifact_plugins()

        artifacts: List[Artifact] = []
        for plugin in plugins:
            try:
                generator = plugin.create_generator(self._chat_client)
                artifact = generator.generate(analysis)
                artifacts.append(artifact)
                logger.info(f"Generated artifact: {plugin.name}")
            except Exception as e:
                logger.error(f"Failed to generate {plugin.name}: {e}")

        logger.info(f"Generated {len(artifacts)} artifacts")
        return artifacts

    def generate_structural_artifacts(self) -> List[Artifact]:
        """
        Generate only structural artifacts (no LLM required).

        Useful for quick generation without API costs.
        """
        if not self.artifacts_enabled:
            return []

        analysis = self.analyze_project()
        plugins = [p for p in self.get_applicable_artifact_plugins() if not p.requires_llm]

        artifacts: List[Artifact] = []
        for plugin in plugins:
            try:
                generator = plugin.create_generator()
                artifact = generator.generate(analysis)
                artifacts.append(artifact)
                logger.info(f"Generated artifact: {plugin.name}")
            except Exception as e:
                logger.error(f"Failed to generate {plugin.name}: {e}")

        return artifacts

    def save_cache(self) -> None:
        """Save the summary cache to disk."""
        if self._summarizer:
            self._summarizer.save_cache()


__all__ = [
    "EnrichmentPipeline",
    "ChatClient",
]
