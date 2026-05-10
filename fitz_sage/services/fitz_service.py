# fitz_sage/services/fitz_service.py
"""
FitzService - Unified service layer for all Fitz operations.

This is THE single API that CLI, SDK, and REST API should all use.
By centralizing business logic here, we:
1. Test once, confidence in all interfaces
2. Eliminate code duplication across CLI/SDK/API
3. Ensure consistent behavior everywhere

Design Principles:
- Stateless: No instance state, all state passed as parameters
- Synchronous: Async wrappers added by callers (API)
- Config-driven: FitzConfig passed to operations that need it
- Exception-based: Raises domain exceptions, interfaces translate

Usage:
    from fitz_sage.services import FitzService

    service = FitzService()

    # Point at docs (progressive querying)
    manifest = service.point("/path/to/docs", collection="docs")

    # Query
    answer = service.query("What is RAG?", collection="docs")

    # Collections
    collections = service.list_collections()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.core import Answer
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.retrieval.rewriter.types import ConversationContext

logger = get_logger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class CollectionInfo:
    """Information about a collection."""

    name: str
    chunk_count: int
    vector_dimensions: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    valid: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Result of system health check."""

    healthy: bool
    components: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


# =============================================================================
# Exceptions
# =============================================================================


class FitzServiceError(Exception):
    """Base exception for service errors."""

    pass


class CollectionNotFoundError(FitzServiceError):
    """Collection does not exist."""

    def __init__(self, collection: str):
        self.collection = collection
        super().__init__(f"Collection not found: {collection}")


class ConfigurationError(FitzServiceError):
    """Configuration is invalid or missing."""

    pass


class QueryError(FitzServiceError):
    """Query failed."""

    pass


# =============================================================================
# FitzService
# =============================================================================


class FitzService:
    """
    Unified service layer for all Fitz operations.

    This class is the single source of truth for business logic.
    CLI, SDK, and API should all call these methods.

    The service is stateless - configuration and collection are passed
    to each method that needs them.
    """

    # =========================================================================
    # Query Operations
    # =========================================================================

    def query(
        self,
        question: str,
        collection: str,
        *,
        top_k: int | None = None,
        conversation_context: "ConversationContext | None" = None,
        engine: str | None = None,
    ) -> Answer:
        """
        Query the knowledge base.

        Args:
            question: The question to ask
            collection: Collection to query
            top_k: Number of results to retrieve (uses config default if None)
            conversation_context: For query rewriting (pronoun resolution)
            engine: Engine to use (None = user's default engine)

        Returns:
            Answer with text, provenance, and mode

        Raises:
            QueryError: If query fails
            CollectionNotFoundError: If collection doesn't exist
        """
        from fitz_sage.core import Query
        from fitz_sage.runtime import create_engine

        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        try:
            engine_instance = create_engine(engine)
            engine_instance.load(collection)

            metadata: dict[str, Any] = {}
            if conversation_context is not None:
                metadata["conversation_context"] = conversation_context

            query_obj = Query(text=question, metadata=metadata)
            return engine_instance.answer(query_obj)

        except Exception as e:
            logger.error("Query failed", error=str(e), collection=collection, exc_info=True)
            raise QueryError(f"Query failed: {e}") from e

    # =========================================================================
    # Point Operations
    # =========================================================================

    def point(
        self,
        source: str | Path,
        collection: str,
        *,
        start_worker: bool = True,
    ) -> Any:
        """Point at a source directory for progressive querying.

        Builds manifest, returns immediately. Queries work instantly via
        agentic search; progressively faster as background indexing completes.

        Args:
            source: Path to file or directory
            collection: Target collection name
            start_worker: Whether to start background indexing thread.
                         False for short-lived CLI processes, True for SDK/API.

        Returns:
            FileManifest with registered files

        Raises:
            ValueError: If source doesn't exist
        """
        from fitz_sage.runtime import create_engine

        source_path = Path(source)
        # Resolve to absolute path to prevent path traversal
        try:
            source_path = source_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid source path: {source}") from e

        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        engine = create_engine()
        engine.load(collection)
        return engine.point(source_path, collection, start_worker=start_worker)

    # =========================================================================
    # Collection Operations
    # =========================================================================

    def list_collections(self) -> list[CollectionInfo]:
        """
        List all collections.

        Returns:
            List of CollectionInfo with names and stats
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from fitz_sage.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        if not hasattr(vdb, "list_collections"):
            return []

        names = vdb.list_collections()
        if not names:
            return []

        # Check if vector DB supports batch stats (most efficient)
        if hasattr(vdb, "batch_get_collection_stats"):
            try:
                # Best case: single query for all stats
                all_stats = vdb.batch_get_collection_stats(names)
                result = []
                for name in names:
                    stats = all_stats.get(name, {})
                    result.append(
                        CollectionInfo(
                            name=name,
                            chunk_count=stats.get("chunk_count", 0),
                            vector_dimensions=stats.get("vector_size"),
                            metadata=stats,
                        )
                    )
                return result
            except Exception as e:
                logger.warning("Batch stats failed, falling back to parallel fetch", error=str(e))
                # Fall through to parallel fetching

        # Fallback: Use parallel fetching to avoid N+1 performance issue
        # For 100 collections: sequential = ~10s, parallel = ~0.5s
        def fetch_collection_info(name: str) -> CollectionInfo:
            """Fetch info for a single collection."""
            info = CollectionInfo(name=name, chunk_count=0)

            if hasattr(vdb, "count"):
                try:
                    info.chunk_count = vdb.count(name)
                except Exception as e:
                    logger.warning("Failed to get chunk count", collection=name, error=str(e))
                    # Continue with default count of 0

            if hasattr(vdb, "get_collection_stats"):
                try:
                    stats = vdb.get_collection_stats(name)
                    info.vector_dimensions = stats.get("vector_size")
                    info.metadata = stats
                except Exception as e:
                    logger.warning("Failed to get collection stats", collection=name, error=str(e))
                    # Continue with default stats

            return info

        # Fetch all collection info in parallel
        # Limit workers to avoid overwhelming the database
        max_workers = min(10, len(names))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_name = {executor.submit(fetch_collection_info, name): name for name in names}

            # Collect results maintaining order
            result = []
            for future in as_completed(future_to_name):
                try:
                    info = future.result()
                    result.append(info)
                except Exception as e:
                    name = future_to_name[future]
                    logger.error(f"Failed to get info for collection '{name}': {e}")
                    # Add minimal info on failure
                    result.append(CollectionInfo(name=name, chunk_count=0))

        # Sort by name for consistent ordering
        result.sort(key=lambda x: x.name)

        return result

    def get_collection(self, name: str) -> CollectionInfo:
        """
        Get detailed information about a collection.

        Args:
            name: Collection name

        Returns:
            CollectionInfo with full stats

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """
        if not self._collection_exists(name):
            raise CollectionNotFoundError(name)

        from fitz_sage.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        info = CollectionInfo(name=name, chunk_count=0)

        if hasattr(vdb, "count"):
            info.chunk_count = vdb.count(name)

        if hasattr(vdb, "get_collection_stats"):
            stats = vdb.get_collection_stats(name)
            info.vector_dimensions = stats.get("vector_size")
            info.metadata = stats

        return info

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Args:
            name: Collection to delete

        Returns:
            True if deleted, False if didn't exist
        """
        from fitz_sage.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        if not hasattr(vdb, "delete_collection"):
            raise FitzServiceError("Vector DB does not support deleting collections")

        if not self._collection_exists(name):
            return False

        vdb.delete_collection(name)
        logger.info("Deleted collection", collection=name)
        return True

    def _collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        from fitz_sage.vector_db.registry import get_vector_db_plugin

        vdb = get_vector_db_plugin()

        if hasattr(vdb, "list_collections"):
            return name in vdb.list_collections()

        # Fallback: try to count
        if hasattr(vdb, "count"):
            try:
                vdb.count(name)
                return True
            except Exception as e:
                logger.debug(f"Collection '{name}' does not exist (count check failed): {e}")
                return False

        return True  # Assume exists if we can't check

    # =========================================================================
    # Configuration Operations
    # =========================================================================

    def validate_config(self) -> ConfigValidationResult:
        """
        Validate the current configuration.

        Checks:
        - Config file exists and parses
        - Required plugins are available
        - API keys are set for configured providers
        - Vector DB is accessible

        Returns:
            ConfigValidationResult with issues and warnings
        """
        issues = []
        warnings = []

        # Check config exists
        from fitz_sage.core.paths import FitzPaths

        config_path = FitzPaths.config()
        if not config_path.exists():
            issues.append(f"Config not found: {config_path}")
            return ConfigValidationResult(valid=False, issues=issues)

        # Try to load
        try:
            from fitz_sage.cli.context import CLIContext

            ctx = CLIContext.load()
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            issues.append(f"Config parse error: {e}")
            return ConfigValidationResult(valid=False, issues=issues)

        # Check plugins
        try:
            from fitz_sage.llm import get_chat_factory

            get_chat_factory(ctx.chat_tier_specs)
        except Exception as e:
            logger.warning(f"Chat plugin '{ctx.chat_plugin}' validation failed: {e}")
            issues.append(f"Chat plugin '{ctx.chat_plugin}' not available: {e}")

        # Check vector DB
        try:
            from fitz_sage.vector_db.registry import get_vector_db_plugin

            get_vector_db_plugin(ctx.vector_db_plugin)
        except Exception as e:
            logger.warning(f"Vector DB '{ctx.vector_db_plugin}' validation failed: {e}")
            issues.append(f"Vector DB '{ctx.vector_db_plugin}' not available: {e}")

        return ConfigValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
        )

    def get_config_summary(self) -> dict[str, Any]:
        """
        Get a summary of current configuration.

        Returns:
            Dict with plugin names, models, and settings
        """
        try:
            from fitz_sage.cli.context import CLIContext

            ctx = CLIContext.load()

            return {
                "chat": ctx.chat_plugin,
                "chat_model_smart": ctx.chat_model_smart,
                "chat_model_fast": ctx.chat_model_fast,
                "rerank": ctx.rerank_plugin,
                "vector_db": ctx.vector_db_plugin,
                "retrieval_plugin": ctx.retrieval_plugin,
                "collection": ctx.retrieval_collection,
                "chunk_size": ctx.chunk_size,
                "chunk_overlap": ctx.chunk_overlap,
            }
        except Exception as e:
            logger.error(f"Failed to get config summary: {e}", exc_info=True)
            return {"error": str(e)}

    # =========================================================================
    # Health & Diagnostics
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """
        Check system health.

        Tests connectivity to:
        - Vector database
        - LLM providers (if configured)

        Returns:
            HealthCheckResult with component status
        """
        components = {}
        issues = []

        # Check vector DB
        try:
            from fitz_sage.vector_db.registry import get_vector_db_plugin

            vdb = get_vector_db_plugin()
            if hasattr(vdb, "list_collections"):
                vdb.list_collections()
            components["vector_db"] = True
        except Exception as e:
            logger.warning(f"Vector DB health check failed: {e}")
            components["vector_db"] = False
            issues.append(f"Vector DB: {e}")

        # Check chat provider
        try:
            from fitz_sage.cli.context import CLIContext
            from fitz_sage.llm import get_chat_factory

            ctx = CLIContext.load()
            get_chat_factory(ctx.chat_tier_specs)  # Verify factory works
            components["chat"] = True
        except Exception as e:
            logger.warning(f"Chat provider health check failed: {e}")
            components["chat"] = False
            issues.append(f"Chat provider: {e}")

        return HealthCheckResult(
            healthy=all(components.values()),
            components=components,
            issues=issues,
        )


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_service: FitzService | None = None


def get_service() -> FitzService:
    """Get the default FitzService instance."""
    global _default_service
    if _default_service is None:
        _default_service = FitzService()
    return _default_service
