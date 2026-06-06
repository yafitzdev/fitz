# fitz_sage/governance/__init__.py
"""
Epistemic governance.

A single classifier decides whether retrieved sources support a confident
answer (TRUSTWORTHY), contradict each other (DISPUTED), or simply don't
contain enough information (ABSTAIN). The standard classifier is
[pyrrho](https://huggingface.co/yafitzdev/pyrrho-nano-g3.1), a multitask
ModernBERT model with governance, query-contract, route/domain, taxonomy,
scalar, and optional release-specific heads running locally on CPU.

Governance is mandatory in the standard product path. The ``governance:``
config key declares which pyrrho classifier to use:

    from fitz_sage.governance import create_governance

    governance = create_governance("pyrrho")
    decision = governance.decide(query, retrieved_contexts)
    # decision.mode in {TRUSTWORTHY, DISPUTED, ABSTAIN}
    # decision.probs is the governance softmax distribution
    # decision.query_contract / route / taxonomy expose head metadata
    # governance.classify_query(query) returns pre-retrieval query signals
    # decision.reason is a one-line human-readable summary

The legacy constraint+sklearn cascade was removed in v0.13.0. The pyrrho
classifier is faster than the cascade it replaced and now also supplies
pre-retrieval query signals used by the retrieval stack.
"""

from __future__ import annotations

from .protocol import EvidenceItem
from .pyrrho import GovernanceDecision, Pyrrho, QueryDecision


def create_governance(spec: str) -> Pyrrho:
    """Build a governance classifier from a config spec.

    Args:
        spec: ``"pyrrho"`` (the default classifier),
            ``"pyrrho/<hf-model-id>"`` (a custom pyrrho fine-tune), or
            ``"pyrrho/<local-package-path>"`` (an unpacked pyrrho release).

    Returns:
        A ``Pyrrho`` instance.

    Raises:
        ValueError: if ``spec`` names an unknown governance provider.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Governance must be 'pyrrho' or 'pyrrho/<hf-model-id>'.")

    provider, _, model = spec.partition("/")
    provider, model = provider.strip(), model.strip()
    if provider == "pyrrho":
        return Pyrrho(model_id=model) if model else Pyrrho()
    raise ValueError(
        f"Unknown governance provider: {provider!r}. "
        f"Supported: 'pyrrho' or 'pyrrho/<hf-model-id>'."
    )


__all__ = [
    "EvidenceItem",
    "GovernanceDecision",
    "Pyrrho",
    "QueryDecision",
    "create_governance",
]
