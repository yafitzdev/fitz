# fitz_sage/governance/__init__.py
"""
Epistemic governance.

A single classifier decides whether retrieved sources support a confident
answer (SUFFICIENT), contradict each other (DISPUTED), or simply don't
contain enough information (INSUFFICIENT). The standard classifier is
[pyrrho](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1), a ModernBERT
classifier with v2 evidence-verdict, failure-mode, retrieval-intent, and
evidence-kind heads.

Governance is mandatory in the standard product path. The ``governance:``
config key declares which pyrrho classifier to use:

    from fitz_sage.governance import create_governance

    governance = create_governance("pyrrho")
    decision = governance.decide(query, retrieved_contexts)
    # decision.mode in {SUFFICIENT, DISPUTED, INSUFFICIENT}
    # decision.probs is the governance softmax distribution
    # v2 verdict, failure, retrieval-intent, and evidence-kind heads expose metadata
    # decision.reason is a one-line human-readable summary

Pyrrho owns v2 query-planning heads and evidence governance. KRAG owns the
retrieval mechanics that consume those planning signals.
"""

from __future__ import annotations

from .protocol import EvidenceItem
from .pyrrho import GovernanceDecision, Pyrrho, PyrrhoQueryPlan


def create_governance(spec: str) -> Pyrrho:
    """Build a governance classifier from a config spec.

    Args:
        spec: ``"pyrrho"`` (the default v2 classifier),
            ``"pyrrho/<hf-model-id>"`` (a Pyrrho package), or
            ``"pyrrho/<local-package-path>"`` (an unpacked Pyrrho package).

    Returns:
        A ``Pyrrho`` instance.

    Raises:
        ValueError: if ``spec`` names an unknown governance provider.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Governance must be 'pyrrho' or 'pyrrho/<package>'.")

    provider, _, model = spec.partition("/")
    provider, model = provider.strip(), model.strip()
    if provider == "pyrrho":
        return Pyrrho(model_id=model) if model else Pyrrho()
    raise ValueError(
        f"Unknown governance provider: {provider!r}. " f"Supported: 'pyrrho' or 'pyrrho/<package>'."
    )


__all__ = [
    "EvidenceItem",
    "GovernanceDecision",
    "Pyrrho",
    "PyrrhoQueryPlan",
    "create_governance",
]
