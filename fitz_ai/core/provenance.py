"""
Provenance - Source attribution for answers.

Provenance provides transparency about where information came from.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Provenance:
    """
    Source attribution for answers.

    Provenance links an answer back to its source material, enabling:
    - Verification: Users can check claims against sources
    - Attribution: Proper credit to source documents
    - Debugging: Understanding what knowledge the engine used

    Examples:
        Minimal provenance:
        >>> prov = Provenance(source_id="doc_123")

        Provenance with excerpt:
        >>> prov = Provenance(
        ...     source_id="research_paper_42",
        ...     excerpt="Quantum computers use qubits instead of classical bits..."
        ... )

        Provenance with metadata:
        >>> prov = Provenance(
        ...     source_id="wiki_quantum",
        ...     excerpt="A qubit can be in superposition...",
        ...     metadata={
        ...         "title": "Quantum Computing",
        ...         "author": "Dr. Smith",
        ...         "url": "https://...",
        ...         "relevance_score": 0.94
        ...     }
        ... )
    """

    source_id: str
    """
    Unique identifier for the source.

    This should be stable and allow the source to be retrieved later.
    Format is engine-specific:
    - Classic RAG: might be "chunk_id" or "doc_id:chunk_idx"
    - CLaRa: might be "document_id"
    - Custom engines: whatever makes sense
    """

    excerpt: Optional[str] = None
    """
    Optional excerpt from the source that was used.

    This provides immediate context about what information was used
    without requiring the user to fetch the full source. Particularly
    useful for chunk-based systems where the relevant content is small.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    """
    Source metadata.

    Common fields might include:
    - title: Document title
    - author: Document author
    - url: Link to original source
    - created_at: Source creation date
    - relevance_score: How relevant this source was to the query
    - position: For chunk-based systems, where in the document

    Engines can include whatever metadata makes sense for their use case.
    """

    def __post_init__(self):
        """Validate provenance after initialization."""
        if not self.source_id:
            raise ValueError("Provenance must have a source_id")
