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
        List all collections by enumerating fitz-sage's PostgreSQL databases.

        Each fitz-sage collection is its own database under
        ``PostgresConnectionManager``. We list ``pg_database`` and report
        the krag_section_index row count per collection.
        """
        cm = self._connection_manager()
        names = _list_collection_databases(cm)
        result: list[CollectionInfo] = []
        for name in names:
            result.append(CollectionInfo(name=name, chunk_count=_collection_chunk_count(cm, name)))
        result.sort(key=lambda x: x.name)
        return result

    def get_collection(self, name: str) -> CollectionInfo:
        """Get info about a collection. Raises CollectionNotFoundError if missing."""
        cm = self._connection_manager()
        if name not in _list_collection_databases(cm):
            raise CollectionNotFoundError(name)
        return CollectionInfo(name=name, chunk_count=_collection_chunk_count(cm, name))

    def delete_collection(self, name: str) -> bool:
        """Drop the collection's PostgreSQL database. Returns True on success.

        Uses the connection manager's base URI to issue ``DROP DATABASE``
        in autocommit mode against the ``postgres`` admin database.
        """
        import psycopg

        cm = self._connection_manager()
        if name not in _list_collection_databases(cm):
            return False

        base_uri = getattr(cm, "_base_uri", None)
        if not base_uri:
            cm.start()
            base_uri = getattr(cm, "_base_uri", None)
        if not base_uri:
            raise FitzServiceError("PostgresConnectionManager has no base URI")

        from urllib.parse import urlparse, urlunparse

        admin_uri = urlunparse(urlparse(base_uri)._replace(path="/postgres"))
        with psycopg.connect(admin_uri, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}"')
        logger.info("Deleted collection", collection=name)
        return True

    @staticmethod
    def _connection_manager() -> Any:
        from fitz_sage.storage.postgres import PostgresConnectionManager

        cm = PostgresConnectionManager.get_instance()
        cm.start()
        return cm

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
        - PostgreSQL connection manager
        - LLM chat provider (if configured)
        """
        components = {}
        issues = []

        # Check Postgres connection manager
        try:
            cm = self._connection_manager()
            _list_collection_databases(cm)
            components["postgres"] = True
        except Exception as e:
            logger.warning(f"Postgres health check failed: {e}")
            components["postgres"] = False
            issues.append(f"Postgres: {e}")

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
# Helpers — collection enumeration via PostgresConnectionManager
# =============================================================================


def _list_collection_databases(cm: Any) -> list[str]:
    """Enumerate fitz-sage collection databases (pg_database minus system DBs)."""
    import psycopg
    from urllib.parse import urlparse, urlunparse

    base_uri = getattr(cm, "_base_uri", None)
    if not base_uri:
        cm.start()
        base_uri = getattr(cm, "_base_uri", None)
    if not base_uri:
        return []
    admin_uri = urlunparse(urlparse(base_uri)._replace(path="/postgres"))
    try:
        with psycopg.connect(admin_uri, autocommit=True) as conn:
            rows = conn.execute(
                "SELECT datname FROM pg_database WHERE datistemplate = false"
            ).fetchall()
    except Exception as e:
        logger.warning(f"Failed to list collection databases: {e}")
        return []
    system = {"postgres", "template0", "template1"}
    return [row[0] for row in rows if row[0] not in system]


def _collection_chunk_count(cm: Any, name: str) -> int:
    """Return krag_section_index row count for a collection (0 if table absent)."""
    try:
        with cm.connection(name) as conn:
            exists = conn.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'krag_section_index'
                )
                """
            ).fetchone()[0]
            if not exists:
                return 0
            return int(conn.execute("SELECT COUNT(*) FROM krag_section_index").fetchone()[0])
    except Exception as e:
        logger.debug(f"Chunk count failed for '{name}': {e}")
        return 0


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
