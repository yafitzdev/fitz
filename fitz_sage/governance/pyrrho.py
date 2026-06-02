# fitz_sage/governance/pyrrho.py
"""
Pyrrho governance backend — single-pass ONNX classifier for
TRUSTWORTHY / DISPUTED / ABSTAIN.

Replaces the constraint+sklearn cascade. One forward pass through an
INT8 ONNX ModernBERT-base fine-tune, no LLM call.

The lazy load + forward pass are inherited from `OnnxEncoderBackend`:
the pre-quantized ONNX is pulled straight from the model repo with
`huggingface_hub`, and the tokenizer comes from `transformers`. No
`optimum`, and therefore no `torch`, on the dependency path.

Model card:
    https://huggingface.co/yafitzdev/pyrrho-nano-g3

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
DISPUTED. This is the rule used for the pyrrho-nano-g3 held-out
headline numbers in the model card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.encoders.onnx import OnnxEncoderBackend
from fitz_sage.governance.protocol import EvidenceItem

MODEL_ID = "yafitzdev/pyrrho-nano-g3"
ONNX_FILE = "model_quantized.onnx"  # pre-quantized INT8, at the repo root
MAX_LENGTH = 4096
TAU = 0.60

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

    supports_batched_prefixes = False

    def __init__(self, model_id: str = MODEL_ID) -> None:
        super().__init__(model_id=model_id, onnx_file=ONNX_FILE)

    def decide(self, query: str, contexts: list[EvidenceItem]) -> GovernanceDecision:
        """Classify a (query, contexts) pair into one of the three governance modes.

        Args:
            query: the sanitized user question.
            contexts: retrieved evidence items (anything with a `.content` attribute).

        Returns:
            A GovernanceDecision with the selected AnswerMode, the full softmax
            distribution, and a one-line human-readable reason.
        """
        return self._decide_one(query, contexts)

    def decide_many(
        self,
        query: str,
        contexts_by_prefix: list[list[EvidenceItem]],
    ) -> list[GovernanceDecision]:
        """Classify several evidence prefixes.

        The current pyrrho-nano-g3 ONNX export is not batch-safe under
        onnxruntime on Windows, so this API intentionally runs each prefix as a
        single forward pass.
        """
        return [self._decide_one(query, contexts) for contexts in contexts_by_prefix]

    def _decide_one(self, query: str, contexts: list[EvidenceItem]) -> GovernanceDecision:
        """Classify one evidence prefix."""
        import numpy as np

        if not contexts:
            return GovernanceDecision(
                mode=AnswerMode.ABSTAIN,
                probs=(1.0, 0.0, 0.0),
                reason="Pyrrho: no contexts retrieved.",
            )

        text = _format_input(query, (context.content for context in contexts))
        enc = self._encode(text, truncation=True, max_length=MAX_LENGTH)
        logits = self._run(enc)[0]
        return _decision_from_logits(logits, np)


def _decision_from_logits(logits: Any, np: Any) -> GovernanceDecision:
    """Convert one logits row into a calibrated Pyrrho decision."""
    probs_arr = _softmax(logits, np)
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


def _softmax(logits: Any, np: Any) -> Any:
    """Softmax one logits row with numerical stabilization."""
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


__all__ = ["GovernanceDecision", "Pyrrho", "MODEL_ID", "TAU"]
