# fitz_ai/engines/classic_rag/constraints/runner.py
"""
Constraint Runner - Applies constraint plugins to retrieved context.

This module provides the function that orchestrates constraint evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from .base import ConstraintPlugin, ConstraintResult

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk

logger = get_logger(__name__)


def apply_constraints(
    query: str,
    chunks: Sequence["Chunk"],
    constraints: Sequence[ConstraintPlugin],
) -> ConstraintResult:
    """
    Apply all constraints to retrieved context.

    If ANY constraint denies a decisive answer, the combined result denies.
    This is fail-safe: constraints only restrict, never expand.

    Args:
        query: The user's question
        chunks: Retrieved chunks (post-retrieval, pre-generation)
        constraints: List of constraint plugins to apply

    Returns:
        Combined ConstraintResult (deny if any constraint denies)
    """
    if not constraints:
        return ConstraintResult.allow()

    logger.debug(f"{PIPELINE} Applying {len(constraints)} constraint(s)")

    denial_reasons: list[str] = []
    all_metadata: dict = {}

    for constraint in constraints:
        try:
            result = constraint.apply(query, chunks)

            if not result.allow_decisive_answer:
                logger.info(
                    f"{PIPELINE} Constraint '{constraint.name}' denied: {result.reason}"
                )
                if result.reason:
                    denial_reasons.append(result.reason)
                all_metadata[constraint.name] = result.metadata

        except Exception as e:
            # Fail-safe: if constraint crashes, log and continue
            # Do NOT block the answer due to constraint errors
            logger.warning(
                f"{PIPELINE} Constraint '{constraint.name}' raised exception: {e}"
            )
            continue

    if denial_reasons:
        combined_reason = "; ".join(denial_reasons)
        return ConstraintResult.deny(
            reason=combined_reason,
            constraint_results=all_metadata,
        )

    logger.debug(f"{PIPELINE} All constraints passed")
    return ConstraintResult.allow()
