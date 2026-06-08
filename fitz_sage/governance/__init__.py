# fitz_sage/governance/__init__.py
"""
Epistemic governance.

A single classifier decides whether retrieved sources support a confident
answer (TRUSTWORTHY), contradict each other (DISPUTED), or simply don't
contain enough information (ABSTAIN). The standard classifier is
[pyrrho](https://huggingface.co/yafitzdev/pyrrho-nano-g4-alpha), a multitask
ModernBERT model with governance, query-contract, route/domain, taxonomy,
retrieval-action, gap-type, answerability-shape, retrieval-modality, and scalar
heads running locally on CPU.

Governance is mandatory in the standard product path. The ``governance:``
config key declares which pyrrho classifier to use:

    from fitz_sage.governance import create_governance

    governance = create_governance("pyrrho")
    decision = governance.decide(query, retrieved_contexts)
    # decision.mode in {TRUSTWORTHY, DISPUTED, ABSTAIN}
    # decision.probs is the governance softmax distribution
    # decision query, evidence, route, taxonomy, action, gap, modality heads expose metadata
    # governance.classify_query(query) returns pre-retrieval query signals
    # decision.reason is a one-line human-readable summary

Pyrrho now supplies both evidence governance and pre-retrieval query signals
used by the retrieval stack.
"""

from __future__ import annotations

from .evidence_contract import QueryContract, build_query_contract
from .protocol import EvidenceItem
from .pyrrho import GovernanceDecision, Pyrrho, QueryDecision


def create_governance(spec: str) -> Pyrrho:
    """Build a governance classifier from a config spec.

    Args:
        spec: ``"pyrrho"`` (the default g4-alpha classifier),
            ``"pyrrho/<hf-model-id>"`` (a compatible Pyrrho g4 package), or
            ``"pyrrho/<local-package-path>"`` (an unpacked compatible package).

    Returns:
        A ``Pyrrho`` instance.

    Raises:
        ValueError: if ``spec`` names an unknown governance provider.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(
            "Governance must be 'pyrrho' or 'pyrrho/<compatible-g4-package>'."
        )

    provider, _, model = spec.partition("/")
    provider, model = provider.strip(), model.strip()
    if provider == "pyrrho":
        return Pyrrho(model_id=model) if model else Pyrrho()
    raise ValueError(
        f"Unknown governance provider: {provider!r}. "
        f"Supported: 'pyrrho' or 'pyrrho/<compatible-g4-package>'."
    )


__all__ = [
    "EvidenceItem",
    "GovernanceDecision",
    "Pyrrho",
    "QueryContract",
    "QueryDecision",
    "build_query_contract",
    "create_governance",
]
