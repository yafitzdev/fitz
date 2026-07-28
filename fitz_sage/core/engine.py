# fitz_sage/core/engine.py
"""Knowledge engine protocols. See docs/API_REFERENCE.md for details."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .answer import Answer
from .evidence import EvidencePack
from .query import Query
from .retrieval_run import RetrievalRun


@runtime_checkable
class KnowledgeEngine(Protocol):
    """Paradigm-agnostic protocol: all engines implement answer(Query) -> Answer."""

    def answer(self, query: Query) -> Answer:
        """Execute query and return answer with provenance."""
        ...


@runtime_checkable
class RetrievalEngine(KnowledgeEngine, Protocol):
    """A KnowledgeEngine that ingests sources and exposes retrieval evidence.

    Engines that support persistent ingest (see EngineCapabilities) implement
    this richer contract. Typical lifecycle::

        engine = create_engine("fitz_krag")
        engine.load("my_docs")              # bind to a collection
        engine.point(Path("./docs"))        # returns with searchable source index
        engine.wait_for_enrichment()        # optional: block on derived metadata
        evidence = engine.evidence(Query(text="...?")) # governed evidence
        sources = engine.retrieve(Query(text="...?"))  # raw sources, no synthesis
        answer = engine.answer(Query(text="...?"))     # optional synthesis

    ``retrieve()`` returns a list of engine-specific source objects (for KRAG,
    ``ReadResult`` — content + file_path + line_range).
    """

    def load(self, collection: str) -> None:
        """Bind the engine to a collection."""
        ...

    def point(
        self,
        source: Path,
        collection: str | None = None,
        *,
        start_worker: bool = True,
    ) -> Any:
        """Create a searchable source index and start background enrichment."""
        ...

    def wait_for_enrichment(self) -> None:
        """Optionally block until model-backed background enrichment settles."""
        ...

    def indexing_status(self) -> dict:
        """Report source-index health and enrichment progress."""
        ...

    def retrieve(self, query: Query) -> list:
        """Return the raw retrieved sources behind answer(), without synthesis."""
        ...

    def evidence(self, query: Query) -> EvidencePack:
        """Return governed, serializable evidence behind answer(), without synthesis."""
        ...

    def trace(self, query: Query) -> RetrievalRun:
        """Return a versioned execution record for governed retrieval."""
        ...
