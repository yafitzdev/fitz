# fitz_sage/core/engine.py
"""Knowledge engine protocols. See docs/API_REFERENCE.md for details."""

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
    """A KnowledgeEngine that ingests sources and exposes retrieval evidence.

    Engines that support persistent ingest (see EngineCapabilities) implement
    this richer contract. Typical lifecycle::

        engine = create_engine("fitz_krag")
        engine.load("my_docs")              # bind to a collection
        engine.point(Path("./docs"))        # register a source
        engine.wait_for_query_surface()     # parsed units are searchable
        engine.wait_for_indexing()          # optional: block through keywording
        evidence = engine.evidence(Query(text="...?")) # governed evidence
        sources = engine.retrieve(Query(text="...?"))  # raw sources, no synthesis
        answer = engine.answer(Query(text="...?"))     # optional synthesis

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
        """Block until pointed sources reach query-ready keyword indexing."""
        ...

    def wait_for_query_surface(self) -> None:
        """Block until parsed source units are searchable."""
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
