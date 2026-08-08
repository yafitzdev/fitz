# fitz_sage/services/fitz_service.py
"""
FitzService - Unified service layer for all Fitz operations.

This is THE single API that CLI, SDK, and REST API should all use.
By centralizing business logic here, we:
1. Test once, confidence in all interfaces
2. Eliminate code duplication across CLI/SDK/API
3. Ensure consistent behavior everywhere

Design Principles:
- Collection-scoped: Engine instances are cached by engine and collection
- Synchronous: Async wrappers added by callers (API)
- Config-driven: FitzConfig passed to operations that need it
- Exception-based: Raises domain exceptions, interfaces translate

Usage:
    from fitz_sage.services import FitzService

    service = FitzService()

    # Point at docs (searchable when this returns)
    manifest = service.point("/path/to/docs", collection="docs")

    # Optional answer synthesis
    answer = service.answer("What is RAG?", collection="docs")

    # Collections
    collections = service.list_collections()
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, RLock
from typing import TYPE_CHECKING, Any, Iterator, cast

from fitz_sage.core import Answer, EvidencePack, RetrievalRun
from fitz_sage.core.collections import validate_collection_name
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.core.engine import RetrievalEngine
    from fitz_sage.retrieval.rewriter.types import ConversationContext

logger = get_logger(__name__)


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class CollectionInfo:
    """Information about a collection."""

    name: str
    item_count: int
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


@dataclass
class _EngineSlot:
    """Coordinate concurrent reads and exclusive mutations for one engine."""

    engine: Any
    _condition: Condition = field(default_factory=Condition, init=False, repr=False)
    _readers: int = field(default=0, init=False, repr=False)
    _writer: bool = field(default=False, init=False, repr=False)
    _waiting_writers: int = field(default=0, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)

    def acquire(self, *, exclusive: bool) -> None:
        """Acquire a shared query lease or an exclusive mutation lease."""
        with self._condition:
            if exclusive:
                self._waiting_writers += 1
                try:
                    self._condition.wait_for(
                        lambda: self._closing or (not self._writer and self._readers == 0)
                    )
                finally:
                    self._waiting_writers -= 1
            else:
                self._condition.wait_for(
                    lambda: self._closing or (not self._writer and self._waiting_writers == 0)
                )

            if self._closing:
                raise FitzServiceError("Collection engine is closing")
            if exclusive:
                self._writer = True
            else:
                self._readers += 1

    def release(self, *, exclusive: bool) -> None:
        """Release a previously acquired lease."""
        with self._condition:
            if exclusive:
                if not self._writer:
                    raise RuntimeError("Engine mutation lease is not active")
                self._writer = False
            else:
                if self._readers == 0:
                    raise RuntimeError("Engine query lease is not active")
                self._readers -= 1
            self._condition.notify_all()

    def close(self) -> None:
        """Reject new work and wait for active calls to finish."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: not self._writer and self._readers == 0)


# =============================================================================
# FitzService
# =============================================================================


class FitzService:
    """
    Unified service layer for all Fitz operations.

    This class is the single source of truth for business logic.
    CLI, SDK, and API should all call these methods.

    Engine instances are cached per engine and collection. Queries share an
    engine concurrently; collection mutations remain exclusive.
    """

    def __init__(self) -> None:
        self._engines: dict[tuple[str, str], _EngineSlot] = {}
        self._deleting_collections: set[str] = set()
        self._cache_lock = RLock()

    # =========================================================================
    # Answer Operations
    # =========================================================================

    def answer(
        self,
        question: str,
        collection: str,
        *,
        conversation_context: "ConversationContext | None" = None,
        engine: str | None = None,
    ) -> Answer:
        """
        Synthesize an answer from retrieved evidence.

        Args:
            question: The question to ask
            collection: Collection to query
            conversation_context: For query rewriting (pronoun resolution)
            engine: Engine to use (None = user's default engine)

        Returns:
            Answer with text, provenance, and mode

        Raises:
            QueryError: If query fails
            CollectionNotFoundError: If collection doesn't exist
        """
        from fitz_sage.core import Query

        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        try:
            metadata = self._query_metadata(conversation_context)
            with self._engine(collection, engine) as engine_instance:
                return cast(Answer, engine_instance.answer(Query(text=question, metadata=metadata)))

        except Exception as e:
            logger.error(f"Answer failed (collection={collection}): {e}", exc_info=True)
            raise QueryError(f"Answer failed: {e}") from e

    def evidence(
        self,
        question: str,
        collection: str,
        *,
        conversation_context: "ConversationContext | None" = None,
        engine: str | None = None,
    ) -> EvidencePack:
        """
        Retrieve governed evidence without answer synthesis.

        Args:
            question: The question to retrieve evidence for
            collection: Collection to query
            conversation_context: For query rewriting (pronoun resolution)
            engine: Engine to use (None = user's default engine)

        Returns:
            EvidencePack with ranked source units and governance metadata

        Raises:
            QueryError: If evidence retrieval fails
        """
        from fitz_sage.core import Query

        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        try:
            metadata = self._query_metadata(conversation_context)
            with self._engine(collection, engine) as engine_instance:
                return cast(
                    EvidencePack,
                    engine_instance.evidence(Query(text=question, metadata=metadata)),
                )

        except Exception as e:
            logger.error(f"Evidence retrieval failed (collection={collection}): {e}", exc_info=True)
            raise QueryError(f"Evidence retrieval failed: {e}") from e

    def trace(
        self,
        question: str,
        collection: str,
        *,
        conversation_context: "ConversationContext | None" = None,
        engine: str | None = None,
    ) -> RetrievalRun:
        """Execute governed retrieval and return its versioned execution record."""
        from fitz_sage.core import Query

        if not question or not question.strip():
            raise QueryError("Question cannot be empty")

        try:
            metadata = self._query_metadata(conversation_context)
            with self._engine(collection, engine) as engine_instance:
                return cast(
                    RetrievalRun,
                    engine_instance.trace(Query(text=question, metadata=metadata)),
                )
        except Exception as e:
            logger.error(
                f"Retrieval trace failed (collection={collection}): {e}",
                exc_info=True,
            )
            raise QueryError(f"Retrieval trace failed: {e}") from e

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
        """Build a searchable source index and start background enrichment.

        Args:
            source: Path to file or directory
            collection: Target collection name
            start_worker: Whether to start the background enrichment thread.
                         False for short-lived CLI processes, True for SDK/API.

        Returns:
            FileManifest with indexed files

        Raises:
            ValueError: If source doesn't exist
        """
        source_path = Path(source)
        # Resolve to absolute path to prevent path traversal
        try:
            source_path = source_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid source path: {source}") from e

        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source_path}")

        collection = validate_collection_name(collection)
        with self._engine(collection, exclusive=True) as engine_instance:
            return engine_instance.point(
                source_path,
                collection,
                start_worker=start_worker,
            )

    def indexing_status(self, collection: str) -> dict:
        """Report source-index health and enrichment progress for a collection.

        Loads the persisted manifest (via a fresh engine), so it reflects
        progress made by the background worker across processes.
        """
        with self._engine(collection) as engine_instance:
            return cast(dict[Any, Any], engine_instance.indexing_status())

    # =========================================================================
    # Collection Operations
    # =========================================================================

    def list_collections(self) -> list[CollectionInfo]:
        """List collections by scanning the SQLite storage directory.

        Each fitz-sage collection is its own ``.db`` file under the
        storage directory; the row count comes from the file's
        ``krag_section_index`` table when present.
        """
        cm = self._connection_manager()
        names = cm.list_collections()
        result: list[CollectionInfo] = []
        for name in names:
            result.append(CollectionInfo(name=name, item_count=_collection_item_count(cm, name)))
        result.sort(key=lambda x: x.name)
        return result

    def get_collection(self, name: str) -> CollectionInfo:
        """Get info about a collection. Raises CollectionNotFoundError if missing."""
        name = validate_collection_name(name)
        cm = self._connection_manager()
        if name not in cm.list_collections():
            raise CollectionNotFoundError(name)
        return CollectionInfo(name=name, item_count=_collection_item_count(cm, name))

    def delete_collection(self, name: str) -> bool:
        """Delete a collection's database and persisted manifest state."""
        name = validate_collection_name(name)
        with self._cache_lock:
            if name in self._deleting_collections:
                raise FitzServiceError(f"Collection is already being deleted: {name}")
            self._deleting_collections.add(name)
            keys = [key for key in self._engines if key[1] == name]
            slots = [self._engines.pop(key) for key in keys]

        try:
            for slot in slots:
                slot.close()
                stop = getattr(slot.engine, "stop_background_enrichment", None)
                if callable(stop):
                    stop()

            from fitz_sage.core.paths import FitzPaths
            from fitz_sage.engines.fitz_krag.progressive.write_lock import (
                CollectionWriteLock,
            )

            collections_root = (FitzPaths.workspace() / "collections").resolve()
            collection_dir = (collections_root / name).resolve()
            if collection_dir.parent != collections_root:
                raise FitzServiceError(f"Invalid collection state path: {collection_dir}")

            # Refuse to race a detached indexing/enrichment writer. Releasing
            # before removal is necessary on Windows because the lock file is
            # inside the directory being deleted.
            if collection_dir.exists():
                write_lock = CollectionWriteLock(
                    collection_dir,
                    collection=name,
                    operation="collection deletion",
                )
                write_lock.acquire()
                write_lock.release()

            cm = self._connection_manager()
            database_deleted = bool(cm.delete_collection(name))
            state_deleted = collection_dir.exists()
            if state_deleted:
                shutil.rmtree(collection_dir)

            deleted = database_deleted or state_deleted
            if deleted:
                logger.info(f"Deleted collection: {name}")
            return deleted
        finally:
            with self._cache_lock:
                self._deleting_collections.remove(name)

    @staticmethod
    def _connection_manager() -> Any:
        from fitz_sage.storage.sqlite import SqliteConnectionManager

        cm = SqliteConnectionManager.get_instance()
        cm.start()
        return cm

    @staticmethod
    def _query_metadata(conversation_context: Any | None) -> dict[str, Any]:
        if conversation_context is None:
            return {}
        return {"conversation_context": conversation_context}

    @contextmanager
    def _engine(
        self,
        collection: str,
        engine: str | None = None,
        *,
        exclusive: bool = False,
    ) -> Iterator["RetrievalEngine"]:
        """Yield a cached engine under a shared or exclusive lifecycle lease."""
        from fitz_sage.runtime import create_engine
        from fitz_sage.runtime.registry import get_default_engine

        collection = validate_collection_name(collection)
        engine_name = engine or get_default_engine()
        key = (engine_name, collection)
        with self._cache_lock:
            if collection in self._deleting_collections:
                raise FitzServiceError(f"Collection is being deleted: {collection}")
            slot = self._engines.get(key)
            if slot is None:
                engine_instance = cast("RetrievalEngine", create_engine(engine_name))
                engine_instance.load(collection)
                slot = _EngineSlot(engine_instance)
                self._engines[key] = slot

        slot.acquire(exclusive=exclusive)
        try:
            yield cast("RetrievalEngine", slot.engine)
        finally:
            slot.release(exclusive=exclusive)

    # =========================================================================
    # Configuration Operations
    # =========================================================================

    def validate_config(self) -> ConfigValidationResult:
        """
        Validate the current configuration.

        Checks that the workspace config exists and validates against the
        active engine schema. Provider connectivity belongs to the provider
        itself and is exercised when that optional feature is used.

        Returns:
            ConfigValidationResult with issues and warnings
        """
        issues: list[str] = []
        warnings: list[str] = []

        # Check config exists
        from fitz_sage.core.paths import FitzPaths

        config_path = FitzPaths.config()
        if not config_path.exists():
            issues.append(f"Config not found: {config_path}")
            return ConfigValidationResult(valid=False, issues=issues)

        # Try to load
        try:
            from fitz_sage.config.loader import load_engine_config
            from fitz_sage.runtime import get_default_engine

            load_engine_config(get_default_engine())
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            issues.append(f"Config parse error: {e}")
            return ConfigValidationResult(valid=False, issues=issues)

        return ConfigValidationResult(
            valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
        )

    # =========================================================================
    # Health & Diagnostics
    # =========================================================================

    def health_check(self) -> HealthCheckResult:
        """
        Check system health.

        Tests access to the local SQLite storage layer. Optional endpoint
        providers are not contacted by a general health request.
        """
        components = {}
        issues = []

        # Check SQLite storage layer
        try:
            cm = self._connection_manager()
            cm.list_collections()
            components["sqlite"] = True
        except Exception as e:
            logger.warning(f"SQLite health check failed: {e}")
            components["sqlite"] = False
            issues.append(f"SQLite: {e}")

        return HealthCheckResult(
            healthy=all(components.values()),
            components=components,
            issues=issues,
        )


# =============================================================================
# Helpers — collection enumeration
# =============================================================================


def _collection_item_count(cm: Any, name: str) -> int:
    """Return indexed unit count for a collection (0 if the table is absent)."""
    try:
        with cm.connection(name) as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='krag_section_index'"
            ).fetchone()
            if not exists:
                return 0
            return int(conn.execute("SELECT COUNT(*) FROM krag_section_index").fetchone()[0])
    except Exception as e:
        logger.debug(f"Indexed unit count failed for '{name}': {e}")
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
