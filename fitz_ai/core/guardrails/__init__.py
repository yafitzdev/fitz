# fitz_ai/core/guardrails/__init__.py
"""
Epistemic Guardrails - Constraint system for epistemic correctness.

Guardrails inspect retrieved context and determine what conclusions are allowed.
They are orthogonal to retrieval (what's relevant) and generation (how to answer).

This is a core platform capability supporting epistemic honesty across all engines.

All guardrails use semantic embedding similarity for language-agnostic detection.
This means they work across any language supported by the embedding model.

Default guardrails (all enabled by default):
- ConflictAwareConstraint: blocks confident answers when sources contradict
- InsufficientEvidenceConstraint: blocks confident answers without explicit evidence
- CausalAttributionConstraint: prevents implicit causality claims

Usage:
    from fitz_ai.core.guardrails import (
        ConstraintResult,
        ConstraintPlugin,
        SemanticMatcher,
        create_default_constraints,
        apply_constraints,
    )

    # Create semantic matcher with your embedder
    matcher = SemanticMatcher(embedder=my_embedder.embed)

    # Get default constraints
    constraints = create_default_constraints(matcher)

    # Apply to query + chunks
    result = apply_constraints(query, chunks, constraints)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ConstraintPlugin, ConstraintResult
from .plugins.causal_attribution import CausalAttributionConstraint
from .plugins.conflict_aware import ConflictAwareConstraint
from .plugins.insufficient_evidence import InsufficientEvidenceConstraint
from .runner import apply_constraints
from .semantic import SemanticMatcher

if TYPE_CHECKING:
    from .semantic import EmbedderFunc


def create_default_constraints(
    semantic_matcher: SemanticMatcher,
) -> list[ConstraintPlugin]:
    """
    Create the default constraint plugins with a shared semantic matcher.

    All constraints share the same SemanticMatcher instance for efficiency
    (concept vectors are cached and reused).

    Args:
        semantic_matcher: SemanticMatcher instance with configured embedder

    Returns:
        List of default constraint plugins:
        - ConflictAwareConstraint
        - InsufficientEvidenceConstraint
        - CausalAttributionConstraint
    """
    return [
        ConflictAwareConstraint(semantic_matcher=semantic_matcher),
        InsufficientEvidenceConstraint(semantic_matcher=semantic_matcher),
        CausalAttributionConstraint(semantic_matcher=semantic_matcher),
    ]


def create_semantic_matcher(embedder: "EmbedderFunc") -> SemanticMatcher:
    """
    Create a SemanticMatcher with the given embedder function.

    This is a convenience factory for creating a SemanticMatcher.

    Args:
        embedder: Function that converts text to embedding vector.
                  Signature: (text: str) -> list[float]

    Returns:
        Configured SemanticMatcher instance
    """
    return SemanticMatcher(embedder=embedder)


__all__ = [
    # Core types
    "ConstraintResult",
    "ConstraintPlugin",
    "SemanticMatcher",
    # Constraint implementations
    "ConflictAwareConstraint",
    "InsufficientEvidenceConstraint",
    "CausalAttributionConstraint",
    # Functions
    "apply_constraints",
    "create_default_constraints",
    "create_semantic_matcher",
]
