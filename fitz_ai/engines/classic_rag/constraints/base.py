# fitz_ai/engines/classic_rag/constraints/base.py
"""
Constraint Plugin Base - Protocol and result types.

Constraints answer: "Given this context, what conclusions are allowed?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass(frozen=True)
class ConstraintResult:
    """
    Result of applying a constraint to retrieved context.

    Attributes:
        allow_decisive_answer: If False, generation must avoid authoritative conclusions
        reason: Human-readable explanation (shown to user if answer is constrained)
        signal: Epistemic signal for AnswerMode resolution (e.g., "disputed", "abstain")
        metadata: Additional constraint-specific data for debugging/logging
    """

    allow_decisive_answer: bool
    reason: str | None = None
    signal: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls) -> "ConstraintResult":
        """Factory for allowing decisive answers."""
        return cls(allow_decisive_answer=True)

    @classmethod
    def deny(cls, reason: str, signal: str | None = None, **metadata: Any) -> "ConstraintResult":
        """Factory for denying decisive answers."""
        return cls(
            allow_decisive_answer=False,
            reason=reason,
            signal=signal,
            metadata=metadata,
        )


@runtime_checkable
class ConstraintPlugin(Protocol):
    """
    Protocol for constraint plugins.

    Constraints inspect retrieved chunks and determine whether
    the system is allowed to give a decisive answer.

    Implementations must be:
    - Deterministic (same input → same output)
    - Side-effect free
    - Fast (no LLM calls, no network)
    """

    @property
    def name(self) -> str:
        """Unique name for this constraint."""
        ...

    def apply(
        self,
        query: str,
        chunks: Sequence["Chunk"],
    ) -> ConstraintResult:
        """
        Apply the constraint to retrieved context.

        Args:
            query: The user's question
            chunks: Retrieved chunks (post-retrieval, pre-generation)

        Returns:
            ConstraintResult indicating whether decisive answers are allowed
        """
        ...