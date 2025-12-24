# fitz_ai/engines/classic_rag/constraints/__init__.py
"""
Constraint Plugins - Guardrails for epistemic correctness.

Constraints inspect retrieved context and determine what conclusions are allowed.
They are orthogonal to retrieval (what's relevant) and generation (how to answer).

Default constraints (all enabled by default):
- ConflictAwareConstraint: blocks confident answers when sources contradict
- InsufficientEvidenceConstraint: blocks confident answers without explicit evidence
- CausalAttributionConstraint: prevents implicit causality claims

Usage:
    from fitz_ai.engines.classic_rag.constraints import (
        ConstraintResult,
        ConstraintPlugin,
        ConflictAwareConstraint,
        InsufficientEvidenceConstraint,
        CausalAttributionConstraint,
        apply_constraints,
        get_default_constraints,
    )
"""

from .base import ConstraintPlugin, ConstraintResult
from .plugins.causal_attribution import CausalAttributionConstraint
from .plugins.conflict_aware import ConflictAwareConstraint
from .plugins.insufficient_evidence import InsufficientEvidenceConstraint
from .runner import apply_constraints


def get_default_constraints() -> list[ConstraintPlugin]:
    """
    Get the default constraint plugins.

    Returns all default guardrails:
    - ConflictAwareConstraint
    - InsufficientEvidenceConstraint
    - CausalAttributionConstraint
    """
    return [
        ConflictAwareConstraint(),
        InsufficientEvidenceConstraint(),
        CausalAttributionConstraint(),
    ]


__all__ = [
    "ConstraintResult",
    "ConstraintPlugin",
    "ConflictAwareConstraint",
    "InsufficientEvidenceConstraint",
    "CausalAttributionConstraint",
    "apply_constraints",
    "get_default_constraints",
]
