# fitz_sage/core/engine.py
"""KnowledgeEngine protocols - Query -> Answer contract. See docs/API_REFERENCE.md for details."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .answer import Answer
from .evidence import EvidencePack
from .query import Query


@runtime_checkable
class KnowledgeEngine(Protocol):
    """Paradigm-agnostic protocol: all engines implement answer(Query) -> Answer."""

    def answer(self, query: Query) -> Answer:
        """Execute query and return answer with provenance."""
        ...


@runtime_checkable
class RetrievalEngine(KnowledgeEngine, Protocol):
    """A KnowledgeEngine that also ingests sources and exposes raw retrieval.

    Engines that support persistent ingest (see EngineCapabilities) implement
    this richer contract on top of answer(). Typical lifecycle::

        engine = create_engine("fitz_krag")
        engine.load("my_docs")              # bind to a collection
        engine.point(Path("./docs"))        # register a source (indexes in the
                                            # background; queries work immediately)
        engine.wait_for_indexing()          # optional: block until indexing done
        answer = engine.answer(Query(text="...?"))     # synthesized answer
        sources = engine.retrieve(Query(text="...?"))  # raw sources, no synthesis

    ``retrieve()`` returns a list of engine-specific source objects (for KRAG,
    ``ReadResult`` — content + file_path + line_range).
    """

    def load(self, collection: str) -> None:
        """Bind the engine to a collection."""
        ...

    def point(self, source: Path, collection: str | None = None) -> Any:
        """Register a source directory; indexing proceeds in the background."""
        ...

    def wait_for_indexing(self) -> None:
        """Block until background indexing of pointed sources completes."""
        ...

    def indexing_status(self) -> dict:
        """Report background-indexing progress (file counts by state)."""
        ...

    def retrieve(self, query: Query) -> list:
        """Return the raw retrieved sources behind answer(), without synthesis."""
        ...

    def evidence(self, query: Query) -> EvidencePack:
        """Return governed, serializable evidence behind answer(), without synthesis."""
        ...
