# fitz_sage/governance/__init__.py
"""
Epistemic governance.

A single classifier decides whether retrieved sources support a confident
answer (TRUSTWORTHY), contradict each other (DISPUTED), or simply don't
contain enough information (ABSTAIN). The classifier is the
[pyrrho](https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1)
fine-tune of ModernBERT-base, served as INT8 ONNX on CPU.

Public surface:

    from fitz_sage.governance import decide, GovernanceDecision

    decision = decide(query, retrieved_contexts)
    # decision.mode in {TRUSTWORTHY, DISPUTED, ABSTAIN}
    # decision.probs is the full softmax distribution
    # decision.reason is a one-line human-readable summary

The legacy constraint+sklearn cascade was removed in v0.13.0. The pyrrho
classifier is +7.43 pp accuracy, -0.43 pp false-trustworthy, and ~50x
faster than the cascade it replaced.
"""

from __future__ import annotations

from .protocol import EvidenceItem
from .pyrrho import GovernanceDecision, decide

__all__ = [
    "EvidenceItem",
    "GovernanceDecision",
    "decide",
]
