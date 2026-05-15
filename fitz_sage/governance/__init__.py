# fitz_sage/governance/__init__.py
"""
Epistemic governance.

A single classifier decides whether retrieved sources support a confident
answer (TRUSTWORTHY), contradict each other (DISPUTED), or simply don't
contain enough information (ABSTAIN). The classifier is the
[pyrrho](https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1)
fine-tune of ModernBERT-base, served as INT8 ONNX on CPU.

Governance follows the provider-presence pattern — the ``governance:``
config key declares the classifier (or ``null`` to disable it):

    from fitz_sage.governance import create_governance

    governance = create_governance("pyrrho")   # or None to disable
    if governance is not None:
        decision = governance.decide(query, retrieved_contexts)
        # decision.mode in {TRUSTWORTHY, DISPUTED, ABSTAIN}
        # decision.probs is the full softmax distribution
        # decision.reason is a one-line human-readable summary

The legacy constraint+sklearn cascade was removed in v0.13.0. The pyrrho
classifier is +7.43 pp accuracy, -0.43 pp false-trustworthy, and ~50x
faster than the cascade it replaced.
"""

from __future__ import annotations

from .protocol import EvidenceItem
from .pyrrho import GovernanceDecision, Pyrrho


def create_governance(spec: str | None) -> Pyrrho | None:
    """Build a governance classifier from a config spec.

    Args:
        spec: ``"pyrrho"`` (the default classifier), ``"pyrrho/<hf-model-id>"``
            (a custom pyrrho fine-tune), or ``None`` to disable governance.

    Returns:
        A ``Pyrrho`` instance, or ``None`` when ``spec`` is ``None``.

    Raises:
        ValueError: if ``spec`` names an unknown governance provider.
    """
    if spec is None:
        return None
    provider, _, model = spec.partition("/")
    provider, model = provider.strip(), model.strip()
    if provider == "pyrrho":
        return Pyrrho(model_id=model) if model else Pyrrho()
    raise ValueError(
        f"Unknown governance provider: {provider!r}. "
        f"Supported: 'pyrrho' (or 'pyrrho/<hf-model-id>'), or null to disable."
    )


__all__ = [
    "EvidenceItem",
    "GovernanceDecision",
    "Pyrrho",
    "create_governance",
]
