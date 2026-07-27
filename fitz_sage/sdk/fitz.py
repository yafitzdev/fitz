# fitz_sage/sdk/fitz.py
"""
Fitz class - stateful SDK for the Fitz KRAG retrieval engine.

A thin, stateful wrapper around a single engine instance bound to one
collection. It is the complete programmatic front door — point, query,
retrieve, and wait — so consumers never need to drop down to create_engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from fitz_sage.core import (
    Answer,
    ConfigurationError,
    EvidencePack,
    PyrrhoReplay,
    Query,
    QueryError,
    RetrievalRun,
)
from fitz_sage.core.collections import validate_collection_name
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.core import RetrievalEngine
    from fitz_sage.retrieval.rewriter.types import ConversationContext

logger = get_logger(__name__)


class fitz:
    """
    Stateful SDK for governed Fitz retrieval.

    Holds one engine bound to a collection and exposes the full lifecycle::

        f = fitz(collection="docs")
        f.point("./docs")                  # register documents
        f.wait_for_indexing()              # block for required enrichment
        pack = f.evidence("What is X?")    # governed evidence
        answer = f.query("What is X?")     # optional synthesized answer

    Pointing starts indexing; call ``f.wait_for_indexing()`` before querying when
    using separate ``point`` / ``query`` calls. Convenience methods that receive
    ``source=`` block until required enrichment finishes.

    Examples:
        >>> f = fitz()
        >>> pack = f.evidence("What is the refund policy?", source="./docs")
        >>> print(pack.mode)
        >>> for item in pack.items:
        ...     print(item.excerpt)
    """

    def __init__(
        self,
        collection: str = "default",
        config_path: Optional[Union[str, Path]] = None,
        auto_init: bool = True,
    ) -> None:
        """
        Initialize the Fitz SDK.

        Args:
            collection: Collection name. Documents pointed at with this instance
                are stored here and queries run against it.
            config_path: Path to a YAML config. Defaults to ``.fitz/config.yaml``,
                created automatically on first use if missing.
            auto_init: If True, auto-create a default config when none exists;
                if False, raise ConfigurationError.
        """
        self._collection = validate_collection_name(collection)
        self._config_path = Path(config_path) if config_path else None
        self._auto_init = auto_init
        self._engine: Optional["RetrievalEngine"] = None

    @property
    def collection(self) -> str:
        """The collection name."""
        return self._collection

    @property
    def config_path(self) -> Path:
        """Path to the configuration file."""
        if self._config_path:
            return self._config_path
        from fitz_sage.core.paths import FitzPaths

        return FitzPaths.config()

    def point(self, source: Union[str, Path]) -> None:
        """Register a source file or directory for querying.

        Indexing runs in the background; call ``wait_for_indexing()`` to block
        until required enrichment finishes before querying.
        """
        self._get_engine().point(self._resolve_source(source), self._collection)

    def query(
        self,
        question: str,
        source: Optional[Union[str, Path]] = None,
        conversation_context: Optional["ConversationContext"] = None,
    ) -> Answer:
        """
        Query the knowledge base. Optionally point at a source first.

        Args:
            question: The question to ask.
            source: Optional file/directory to register before querying.
            conversation_context: Optional context for query rewriting
                (conversational pronoun resolution).

        Returns:
            Answer with text, provenance, and epistemic mode.

        Raises:
            ConfigurationError: If not configured and auto_init is False.
            QueryError: If the question is empty.
        """
        if not question or not question.strip():
            raise QueryError("Question cannot be empty")
        engine = self._get_engine()
        if source is not None:
            engine.point(self._resolve_source(source), self._collection)
            engine.wait_for_query_surface()
        return engine.answer(Query(text=question, metadata=self._metadata(conversation_context)))

    def retrieve(
        self,
        question: str,
        conversation_context: Optional["ConversationContext"] = None,
    ) -> list:
        """Retrieve the raw sources behind an answer, without synthesis.

        Returns the engine's retrieved sources (for KRAG, ``ReadResult`` objects
        with content, file_path, and line_range) — useful for building your own
        synthesis or citations.
        """
        if not question or not question.strip():
            raise QueryError("Question cannot be empty")
        return self._get_engine().retrieve(
            Query(text=question, metadata=self._metadata(conversation_context))
        )

    def evidence(
        self,
        question: str,
        source: Optional[Union[str, Path]] = None,
        conversation_context: Optional["ConversationContext"] = None,
    ) -> EvidencePack:
        """Retrieve a governed EvidencePack without answer synthesis."""
        if not question or not question.strip():
            raise QueryError("Question cannot be empty")
        engine = self._get_engine()
        if source is not None:
            engine.point(self._resolve_source(source), self._collection)
            engine.wait_for_query_surface()
        return engine.evidence(Query(text=question, metadata=self._metadata(conversation_context)))

    def trace(
        self,
        question: str,
        source: Optional[Union[str, Path]] = None,
        conversation_context: Optional["ConversationContext"] = None,
    ) -> RetrievalRun:
        """Execute governed retrieval and return a versioned execution record."""
        if not question or not question.strip():
            raise QueryError("Question cannot be empty")
        engine = self._get_engine()
        if source is not None:
            engine.point(self._resolve_source(source), self._collection)
            engine.wait_for_query_surface()
        return engine.trace(Query(text=question, metadata=self._metadata(conversation_context)))

    @staticmethod
    def replay_pyrrho(
        run: RetrievalRun | str | Path,
        pyrrho: str | object | None = None,
    ) -> PyrrhoReplay:
        """Replay Pyrrho over the frozen evidence in a content-bearing trace."""
        from fitz_sage.runtime import replay_pyrrho

        return replay_pyrrho(run, pyrrho)

    def wait_for_indexing(self) -> None:
        """Block until background indexing reaches the query-ready keyword phase."""
        self._get_engine().wait_for_indexing()

    def wait_for_query_surface(self) -> None:
        """Block until parsed retrieval units are searchable."""
        self._get_engine().wait_for_query_surface()

    def indexing_status(self) -> dict:
        """Background-indexing progress for this collection.

        Returns query-ready progress plus explicit ``failed_files`` and
        ``unsupported_files`` inventories. ``complete`` is false when any
        supported file failed.
        """
        return self._get_engine().indexing_status()

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _metadata(conversation_context: "ConversationContext | None") -> dict:
        if conversation_context is None:
            return {}
        return {"conversation_context": conversation_context}

    @staticmethod
    def _resolve_source(source: Union[str, Path]) -> Path:
        """Resolve and validate a source path (guards against traversal)."""
        try:
            return Path(source).resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Source path does not exist: {source}") from e

    def _get_engine(self) -> "RetrievalEngine":
        """Lazily create the collection-bound engine, reused across calls."""
        if self._engine is None:
            self._ensure_config()
            from fitz_sage.runtime import create_engine

            engine = create_engine(
                config_path=str(self._config_path) if self._config_path else None
            )
            engine.load(self._collection)
            self._engine = engine
        return self._engine

    def _ensure_config(self) -> None:
        """Ensure a configuration file exists, creating it if needed."""
        if self.config_path.exists():
            return
        if not self._auto_init:
            raise ConfigurationError(
                f"Config file not found: {self.config_path}. "
                f"Create it manually or pass auto_init=True."
            )
        from fitz_sage.config.firstrun import run_firstrun_setup

        if not run_firstrun_setup():
            raise ConfigurationError(f"Could not initialize config: {self.config_path}")
