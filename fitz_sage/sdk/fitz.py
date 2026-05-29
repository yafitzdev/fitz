# fitz_sage/sdk/fitz.py
"""
Fitz class - Stateful SDK for the Fitz KRAG framework.

This is a thin wrapper around FitzService that adds:
- Stateful collection management (remembers collection across calls)
- Auto-initialization of config
- Simplified API for common use cases
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from fitz_sage.core import Answer, ConfigurationError
from fitz_sage.logging.logger import get_logger
from fitz_sage.services import FitzService

if TYPE_CHECKING:
    from fitz_sage.retrieval.rewriter.types import ConversationContext

logger = get_logger(__name__)


class fitz:
    """
    Stateful SDK for the Fitz RAG framework.

    Provides a single-call workflow:
        answer = fitz.query("question?", source="./docs")

    Queries work immediately via agentic LLM-driven search.
    Background indexing runs silently — queries get progressively faster.

    Examples:
        Simple usage:
        >>> f = fitz()
        >>> answer = f.query("What is the refund policy?", source="./docs")
        >>> print(answer.text)

        With collection name:
        >>> f = fitz(collection="physics")
        >>> answer = f.query("Explain entanglement", source="./physics_papers")

        With custom config:
        >>> f = fitz(config_path="my_config.yaml")

        Access provenance:
        >>> for source in answer.provenance:
        ...     print(source.excerpt)
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
            collection: Name for the collection. Documents ingested
                       with this Fitz instance will be stored in this collection.
            config_path: Path to a YAML config file. If not provided, uses
                        the default config at .fitz/config.yaml or creates one.
            auto_init: If True and no config exists, create a default one.
                      If False, raise ConfigurationError when config missing.
        """
        self._collection = collection
        self._config_path = Path(config_path) if config_path else None
        self._auto_init = auto_init

        # Service layer - does all the real work
        self._service = FitzService()

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

    def query(
        self,
        question: str,
        source: Optional[Union[str, Path]] = None,
        conversation_context: Optional["ConversationContext"] = None,
    ) -> Answer:
        """
        Query the knowledge base. Optionally point at a source directory first.

        Args:
            question: The question to ask.
            source: Path to a file or directory. If provided, registers documents
                before querying (equivalent to CLI --source flag).
            conversation_context: Optional ConversationContext for query rewriting.
                Enables conversational pronoun resolution (e.g., "their" → "TechCorp's").

        Returns:
            Answer object with text and provenance.

        Raises:
            ConfigurationError: If not configured.
            QueryError: If query fails or question is empty.
        """
        self._ensure_config()

        if source is not None:
            self._service.point(source=source, collection=self._collection)

        return self._service.query(
            question=question,
            collection=self._collection,
            conversation_context=conversation_context,
        )

    def _ensure_config(self) -> None:
        """Ensure configuration file exists, creating if needed."""
        config_path = self.config_path

        if config_path.exists():
            return

        if not self._auto_init:
            raise ConfigurationError(
                f"Config file not found: {config_path}. "
                f"Create it manually or pass auto_init=True."
            )

        # Auto-detect providers and write config (same logic as CLI first-run)
        from fitz_sage.core.firstrun import run_firstrun_setup

        if not run_firstrun_setup():
            raise ConfigurationError(f"No LLM provider available. Config: {config_path}")
