# fitz_sage/governance/pyrrho.py
"""
Pyrrho governance backend — single-pass ONNX classifier for
TRUSTWORTHY / DISPUTED / ABSTAIN.

Replaces the constraint+sklearn cascade. One forward pass through an
INT8 ONNX ModernBERT-base fine-tune (~30 ms on CPU), no LLM call.

The lazy load + forward pass are inherited from `OnnxEncoderBackend`:
the pre-quantized ONNX is pulled straight from the model repo with
`huggingface_hub`, and the tokenizer comes from `transformers`. No
`optimum`, and therefore no `torch`, on the dependency path.

Model card:
    https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1

The model expects input in the form:

    Question: <query>

    Sources:
    [1] <ctx 1>
    [2] <ctx 2>
    ...

Output labels (in id order):
    0 -> ABSTAIN
    1 -> DISPUTED
    2 -> TRUSTWORTHY

Calibrated decision rule: if argmax is TRUSTWORTHY but
P(TRUSTWORTHY) < TAU, fall back to the runner-up between ABSTAIN and
DISPUTED. This is the rule that produced the 86.13% / 5.27 % FT
headline numbers in the model card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.encoders.onnx import OnnxEncoderBackend
from fitz_sage.governance.protocol import EvidenceItem

MODEL_ID = "yafitzdev/pyrrho-modernbert-base-v1"
ONNX_FILE = "model_quantized.onnx"  # pre-quantized INT8, at the repo root
MAX_LENGTH = 4096
TAU = 0.50

# id -> AnswerMode
LABELS = (AnswerMode.ABSTAIN, AnswerMode.DISPUTED, AnswerMode.TRUSTWORTHY)


@dataclass(frozen=True)
class GovernanceDecision:
    """Pyrrho's verdict on a (query, contexts) pair."""

    mode: AnswerMode
    probs: tuple[float, float, float]  # (abstain, disputed, trustworthy)
    reason: str

    @property
    def reasons(self) -> tuple[str, ...]:
        """Tuple form for downstream consumers (mirrors the old GovernanceLog API)."""
        return (self.reason,) if self.reason else ()


def _format_input(query: str, contexts: Iterable[str]) -> str:
    """Build the prompt-like string the model was trained on."""
    sources = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    return f"Question: {query}\n\nSources:\n{sources}"


def _reason_for(mode: AnswerMode, probs: tuple[float, float, float]) -> str:
    p_a, p_d, p_t = probs
    if mode is AnswerMode.TRUSTWORTHY:
        return f"Pyrrho: sources support a confident answer (P={p_t:.2f})."
    if mode is AnswerMode.DISPUTED:
        return f"Pyrrho: sources disagree on the answer (P={p_d:.2f})."
    return f"Pyrrho: retrieved sources do not contain enough evidence (P={p_a:.2f})."


class Pyrrho(OnnxEncoderBackend):
    """The pyrrho classifier — one INT8 ONNX forward pass per query."""

    def __init__(self) -> None:
        super().__init__(model_id=MODEL_ID, onnx_file=ONNX_FILE)

    def decide(self, query: str, contexts: list[EvidenceItem]) -> GovernanceDecision:
        """Classify a (query, contexts) pair into one of the three governance modes.

        Args:
            query: the sanitized user question.
            contexts: retrieved evidence items (anything with a `.content` attribute).

        Returns:
            A GovernanceDecision with the selected AnswerMode, the full softmax
            distribution, and a one-line human-readable reason.
        """
        if not contexts:
            return GovernanceDecision(
                mode=AnswerMode.ABSTAIN,
                probs=(1.0, 0.0, 0.0),
                reason="Pyrrho: no contexts retrieved.",
            )

        text = _format_input(query, (c.content for c in contexts))
        enc = self._encode(text, truncation=True, max_length=MAX_LENGTH)
        logits = self._run(enc)[0]  # single input -> first (only) row

        exp = np.exp(logits - logits.max())
        probs_arr = exp / exp.sum()
        probs: tuple[float, float, float] = (
            float(probs_arr[0]),
            float(probs_arr[1]),
            float(probs_arr[2]),
        )

        pred = int(probs_arr.argmax())
        # Calibrated fallback: low-confidence TRUSTWORTHY -> runner-up between A/D.
        if pred == 2 and probs[2] < TAU:
            pred = int(np.argmax(probs_arr[:2]))

        mode = LABELS[pred]
        return GovernanceDecision(mode=mode, probs=probs, reason=_reason_for(mode, probs))


# Process-wide singleton — the model loads lazily on the first decide() call.
_pyrrho = Pyrrho()


def decide(query: str, contexts: list[EvidenceItem]) -> GovernanceDecision:
    """Classify a (query, contexts) pair. Thin wrapper over the process-wide
    `Pyrrho` singleton — see `Pyrrho.decide`."""
    return _pyrrho.decide(query, contexts)


__all__ = ["GovernanceDecision", "Pyrrho", "decide", "MODEL_ID", "TAU"]
