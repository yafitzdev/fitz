# fitz_sage/ingestion/enrichment/pipeline.py
"""
Unified enrichment pipeline.

The EnrichmentPipeline is the single entry point for all enrichment operations.
It coordinates:
1. Chunk-level enrichment (summary, keywords, entities) - ALWAYS ON
2. Hierarchical summaries (L1 group + L2 corpus) - ALWAYS ON
3. Project-level artifacts (type-specific plugins)

All retrieval intelligence features are baked in - no configuration needed.

Usage:
    from fitz_sage.ingestion.enrichment import EnrichmentPipeline, EnrichmentConfig
    from fitz_sage.llm import get_chat_factory

    factory = get_chat_factory("cohere")
    pipeline = EnrichmentPipeline.from_config(
        config=EnrichmentConfig.model_validate(config_dict.get("enrichment", {})),
        project_root=Path("/path/to/project"),
        chat_factory=factory,
        collection="my_collection",  # For keyword vocabulary
    )

    result = pipeline.enrich(chunks)
    # result.chunks have metadata attached
    # keywords saved to vocabulary store
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from fitz_sage.ingestion.enrichment.artifacts.analyzer import ProjectAnalyzer
from fitz_sage.ingestion.enrichment.artifacts.base import Artifact, ProjectAnalysis
from fitz_sage.ingestion.enrichment.artifacts.registry import (
    ArtifactPluginInfo,
    ArtifactRegistry,
)
from fitz_sage.ingestion.enrichment.base import ContentType
from fitz_sage.ingestion.enrichment.bus import ChunkEnricher, create_default_enricher
from fitz_sage.ingestion.enrichment.config import EnrichmentConfig
from fitz_sage.ingestion.enrichment.hierarchy.enricher import HierarchyEnricher
from fitz_sage.ingestion.enrichment.models import EnrichmentResult
from fitz_sage.llm.factory import ChatFactory
from fitz_sage.retrieval.entity_graph import EntityGraphStore
from fitz_sage.retrieval.vocabulary import Keyword, VocabularyStore

if TYPE_CHECKING:
    from fitz_sage.core.chunk import Chunk

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """
    Unified enrichment pipeline.

    Coordinates all enrichment operations:
    - Chunk-level: summary, keywords, entities (via ChunkEnricher bus)
    - Hierarchy-level: L1 group summaries, L2 corpus summary
    - Project-level: artifacts

    All retrieval intelligence is baked in - no opt-in configuration.

    Attributes:
        config: Enrichment configuration
        project_root: Root directory of the project
        collection: Collection name for keyword vocabulary
    """

    def __init__(
        self,
        *,
        config: EnrichmentConfig,
        project_root: Path,
        chat_factory: ChatFactory | None = None,
        collection: str | None = None,
    ):
        self.config = config
        self.project_root = Path(project_root).resolve()
        self._chat_factory = chat_factory
        self._collection = collection

        self._chunk_enricher: ChunkEnricher | None = None
        self._hierarchy_enricher: HierarchyEnricher | None = None
        self._vocabulary_store: VocabularyStore | None = None
        self._entity_graph: EntityGraphStore | None = None
        self._artifact_registry = ArtifactRegistry.get_instance()
        self._project_analysis: ProjectAnalysis | None = None

        self._init_chunk_enricher()
        self._init_hierarchy_enricher()
        self._init_vocabulary_store()
        self._init_entity_graph()

    def _init_chunk_enricher(self) -> None:
        """Initialize the chunk enrichment bus (always on)."""
        if self._chat_factory is None:
            logger.warning("[ENRICHMENT] No chat factory provided, chunk enrichment disabled")
            return

        self._chunk_enricher = create_default_enricher(
            self._chat_factory,
            min_batch_content=self.config.min_batch_content,
        )
        logger.info(
            f"[ENRICHMENT] Chunk enricher initialized with modules: "
            f"{[m.name for m in self._chunk_enricher.modules]}"
        )

    def _init_hierarchy_enricher(self) -> None:
        """Initialize the hierarchy enricher (always on)."""
        if self._chat_factory is None:
            logger.warning("[ENRICHMENT] No chat factory provided, hierarchy enrichment disabled")
            return

        self._hierarchy_enricher = HierarchyEnricher(
            config=self.config.hierarchy,
            chat_factory=self._chat_factory,
        )

        if self.config.hierarchy.rules:
            logger.info(
                f"[ENRICHMENT] Hierarchy enricher initialized with "
                f"{len(self.config.hierarchy.rules)} rules"
            )
        else:
            logger.info(
                f"[ENRICHMENT] Hierarchy enricher initialized in simple mode "
                f"(group_by='{self.config.hierarchy.group_by}')"
            )

    def _init_vocabulary_store(self) -> None:
        """Initialize the vocabulary store for keywords."""
        if self._collection:
            self._vocabulary_store = VocabularyStore(collection=self._collection)
            logger.info(
                f"[ENRICHMENT] Vocabulary store initialized for collection '{self._collection}'"
            )

    def _init_entity_graph(self) -> None:
        """Initialize the entity graph store for related chunk discovery."""
        if self._collection:
            self._entity_graph = EntityGraphStore(collection=self._collection)
            logger.info(
                f"[ENRICHMENT] Entity graph initialized for collection '{self._collection}'"
            )

    @classmethod
    def from_config(
        cls,
        config: EnrichmentConfig | Dict[str, Any] | None,
        project_root: Path,
        chat_factory: ChatFactory | None = None,
        collection: str | None = None,
    ) -> "EnrichmentPipeline":
        """
        Create pipeline from config.

        Args:
            config: EnrichmentConfig, dict, or None (uses defaults)
            project_root: Root directory of the project
            chat_factory: Chat factory for per-task tier selection
            collection: Collection name for keyword vocabulary
        """
        if config is None:
            config = EnrichmentConfig()
        elif isinstance(config, dict):
            config = EnrichmentConfig.model_validate(config)

        return cls(
            config=config,
            project_root=project_root,
            chat_factory=chat_factory,
            collection=collection,
        )

    @property
    def chunk_enrichment_enabled(self) -> bool:
        """Check if chunk-level enrichment is available."""
        return self._chunk_enricher is not None

    @property
    def hierarchy_enrichment_enabled(self) -> bool:
        """Check if hierarchy enrichment is available."""
        return self._hierarchy_enricher is not None

    @property
    def artifacts_enabled(self) -> bool:
        """Check if artifacts are enabled (always true, baked in)."""
        return True

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
            if plugin.requires_llm and self._chat_factory is None:
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
                generator = plugin.create_generator(self._chat_factory)
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

    def enrich(self, chunks: List["Chunk"]) -> EnrichmentResult:
        """
        Unified enrichment entry point.

        This is the main "box" interface for enrichment:
        - Input: List of chunks
        - Output: EnrichmentResult with enriched chunks + artifacts

        The method:
        1. Runs chunk-level enrichment (summary, keywords, entities) via enrichment bus
        2. Saves extracted keywords to vocabulary store
        3. Applies hierarchical enrichment (L1 group + L2 corpus summaries)
        4. Generates corpus-level artifacts
        5. Returns unified result

        All enrichment is baked in - no configuration needed.

        Args:
            chunks: List of chunks to enrich

        Returns:
            EnrichmentResult with enriched chunks and artifacts
        """
        # Step 1: Chunk-level enrichment (summary, keywords, entities)
        if self._chunk_enricher:
            logger.info(f"[ENRICHMENT] Running chunk enrichment on {len(chunks)} chunks")
            enrich_result = self._chunk_enricher.enrich(chunks)
            chunks = enrich_result.chunks

            # Step 2: Save keywords to vocabulary store
            if enrich_result.all_keywords and self._vocabulary_store:
                self._save_keywords(enrich_result.all_keywords, len(chunks))

            # Step 2b: Populate entity graph for related chunk discovery
            if self._entity_graph:
                self._populate_entity_graph(chunks)

        # Step 3: Hierarchical enrichment (L1 group + L2 corpus summaries)
        if self._hierarchy_enricher:
            logger.info("[ENRICHMENT] Running hierarchy enrichment")
            chunks = self._hierarchy_enricher.enrich(chunks)

        # Step 4: Generate artifacts
        artifacts = self.generate_artifacts() if self.artifacts_enabled else []

        return EnrichmentResult(chunks=chunks, artifacts=artifacts)

    def _save_keywords(self, keywords: List[str], source_docs: int) -> None:
        """
        Convert extracted keyword strings to Keyword objects and save.

        Args:
            keywords: List of keyword strings extracted by ChunkEnricher
            source_docs: Number of source documents (for metadata)
        """
        if not self._vocabulary_store:
            return

        # Convert strings to Keyword objects
        # Group by detecting category from format
        keyword_objects: List[Keyword] = []
        seen: set[str] = set()

        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in seen:
                continue
            seen.add(kw_lower)

            category = self._detect_keyword_category(kw)
            keyword_objects.append(
                Keyword(
                    id=kw,
                    category=category,
                    match=[kw],  # LLM-extracted, no variations needed
                    occurrences=1,
                    user_defined=False,
                    auto_generated=[kw],
                )
            )

        if keyword_objects:
            self._vocabulary_store.merge_and_save(keyword_objects, source_docs=source_docs)
            logger.info(f"[ENRICHMENT] Saved {len(keyword_objects)} keywords to vocabulary")

    def _populate_entity_graph(self, chunks: List["Chunk"]) -> None:
        """
        Populate the entity graph with entities from enriched chunks.

        Args:
            chunks: Chunks with entities in metadata (from ChunkEnricher)
        """
        if not self._entity_graph:
            return

        entity_count = 0
        for chunk in chunks:
            entities = chunk.metadata.get("entities", [])
            if not entities:
                continue

            # Convert entity dicts to (name, type) tuples
            entity_tuples = [
                (e.get("name", ""), e.get("type", "unknown")) for e in entities if e.get("name")
            ]

            if entity_tuples:
                self._entity_graph.add_chunk_entities(chunk.id, entity_tuples)
                entity_count += len(entity_tuples)

        if entity_count > 0:
            stats = self._entity_graph.stats()
            logger.info(
                f"[ENRICHMENT] Entity graph updated: {stats['entities']} entities, "
                f"{stats['edges']} edges"
            )

    def _detect_keyword_category(self, keyword: str) -> str:
        """Detect category from keyword format."""
        import re

        # Test case patterns
        if re.match(r"^TC[_\-]?\d+$", keyword, re.IGNORECASE):
            return "testcase"

        # Ticket/issue patterns (PREFIX-NUMBER)
        if re.match(r"^[A-Z]{2,5}-\d+$", keyword):
            return "ticket"

        # Version patterns
        if re.match(r"^v?\d+\.\d+(\.\d+)?", keyword, re.IGNORECASE):
            return "version"

        # Function/method names (camelCase or snake_case)
        if re.match(r"^[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*$", keyword):
            return "function"  # camelCase
        if re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$", keyword):
            return "function"  # snake_case

        # Class names (PascalCase)
        if re.match(r"^[A-Z][a-zA-Z0-9]+$", keyword):
            return "class"

        # Constants (SCREAMING_SNAKE_CASE)
        if re.match(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)+$", keyword):
            return "constant"

        # API endpoints
        if keyword.startswith("/"):
            return "endpoint"

        # Error codes
        if re.match(r"^E\d{3,}$", keyword) or re.match(r"^ERR_", keyword):
            return "error_code"

        return "identifier"


__all__ = [
    "EnrichmentPipeline",
    "EnrichmentResult",
]
