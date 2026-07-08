# fitz_sage/governance/pyrrho.py
"""Pyrrho v2 ONNX governance backend."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.core.paths import FitzPaths
from fitz_sage.governance.protocol import EvidenceItem

MODEL_ID = "yafitzdev/pyrrho-v2-nano-g1"
V2_MAX_LENGTH = 2048
TAU = 0.34
ONNX_MODEL_FILENAMES = ("model.onnx", "model_quantized.onnx")
_DEFAULT_ALLOW_PATTERNS = (
    "config.json",
    "manifest.json",
    "model.onnx",
    "model_quantized.onnx",
    "ort_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

LABELS = (AnswerMode.INSUFFICIENT, AnswerMode.DISPUTED, AnswerMode.SUFFICIENT)

_VERDICT_LABELS = ("INSUFFICIENT", "DISPUTED", "SUFFICIENT")
_FAILURE_LABELS = (
    "none",
    "unresolved_conflict",
    "missing_or_incomplete_evidence",
    "wrong_scope_or_version",
    "ambiguous_request",
)
_RETRIEVAL_INTENTS = (
    "needs_lookup",
    "needs_temporal_resolution",
    "needs_comparison_or_set",
    "needs_broad_coverage",
)
_EVIDENCE_KINDS = (
    "needs_text",
    "needs_table_or_record",
    "needs_code_or_symbol",
    "needs_config_or_setting",
    "needs_log_or_run_result",
    "needs_document_layout",
)
_VERDICT_TO_MODE = {
    "INSUFFICIENT": AnswerMode.INSUFFICIENT,
    "DISPUTED": AnswerMode.DISPUTED,
    "SUFFICIENT": AnswerMode.SUFFICIENT,
}
PYRRHO_PRE_TAG = "[PYRRHO_PRE]"
PYRRHO_POST_TAG = "[PYRRHO_POST]"


@dataclass(frozen=True)
class HeadDecision:
    """Prediction metadata for one Pyrrho single-label head."""

    raw_label: str
    final_label: str
    used_threshold_fallback: bool
    threshold: float | None
    confidence: float
    probabilities: dict[str, float]
    runner_up_label: str
    runner_up_probability: float
    margin_to_runner_up: float
    entropy: float


@dataclass(frozen=True)
class MultiLabelDecision:
    """Prediction metadata for one Pyrrho multi-label head."""

    raw_label: str
    final_label: str
    final_labels: tuple[str, ...]
    used_threshold_fallback: bool
    threshold: float | None
    confidence: float
    probabilities: dict[str, float]
    runner_up_label: str
    runner_up_probability: float
    margin_to_runner_up: float
    entropy: float


@dataclass(frozen=True)
class GovernanceDecision:
    """Pyrrho's v2 verdict on a (query, contexts) pair."""

    mode: AnswerMode
    probs: tuple[float, float, float]
    reason: str
    evidence_verdict: HeadDecision | None = None
    failure_mode: HeadDecision | None = None
    retrieval_intents: MultiLabelDecision | None = None
    evidence_kinds: MultiLabelDecision | None = None
    heads: dict[str, HeadDecision | MultiLabelDecision] = field(default_factory=dict)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Tuple form for downstream consumers."""
        return (self.reason,) if self.reason else ()


@dataclass(frozen=True)
class PyrrhoQueryPlan:
    """Pyrrho's v2 pre-retrieval classification for a user query."""

    retrieval_intents: MultiLabelDecision
    evidence_kinds: MultiLabelDecision
    heads: dict[str, MultiLabelDecision] = field(default_factory=dict)


def _format_query_input(query: str) -> str:
    """Build the query-only text the dual Pyrrho model was trained on."""
    return f"{PYRRHO_PRE_TAG}\nQuestion: {query}"


def _format_input(query: str, contexts: Iterable[str]) -> str:
    """Build the evidence-conditioned text the model was trained on."""
    sources = "\n".join(f"[{i}] {context}" for i, context in enumerate(contexts, start=1))
    return f"{PYRRHO_POST_TAG}\nQuestion: {query}\n\nSources:\n{sources}"


def _reason_for(mode: AnswerMode, probs: tuple[float, float, float]) -> str:
    p_a, p_d, p_t = probs
    if mode is AnswerMode.SUFFICIENT:
        return f"Pyrrho: evidence is sufficient for a confident answer (P={p_t:.2f})."
    if mode is AnswerMode.DISPUTED:
        return f"Pyrrho: evidence is disputed (P={p_d:.2f})."
    return f"Pyrrho: evidence is insufficient (P={p_a:.2f})."


class Pyrrho:
    """The Pyrrho v2 evidence governance classifier."""

    supports_batched_prefixes = True

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._model_dir: Path | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._sufficient_threshold = TAU

    def decide(self, query: str, contexts: list[EvidenceItem]) -> GovernanceDecision:
        """Classify one (query, contexts) pair into a governance mode."""
        return self.decide_many(query, [contexts])[0]

    def plan_query(self, query: str) -> PyrrhoQueryPlan:
        """Classify one query before retrieval into native v2 planning heads."""
        self._load()
        logits = self._run_onnx_texts([_format_query_input(query)])[0]
        _, _, retrieval_intents, evidence_kinds = _v2_core_heads(
            logits,
            sufficient_threshold=None,
        )
        return PyrrhoQueryPlan(
            retrieval_intents=retrieval_intents,
            evidence_kinds=evidence_kinds,
            heads={
                "retrieval_intents": retrieval_intents,
                "evidence_kinds": evidence_kinds,
            },
        )

    def decide_many(
        self,
        query: str,
        contexts_by_prefix: list[list[EvidenceItem]],
    ) -> list[GovernanceDecision]:
        """Classify several evidence prefixes in one model batch."""
        decisions: list[GovernanceDecision | None] = [None] * len(contexts_by_prefix)
        non_empty_contexts: list[list[EvidenceItem]] = []
        non_empty_positions: list[int] = []

        for index, contexts in enumerate(contexts_by_prefix):
            if not contexts:
                decisions[index] = GovernanceDecision(
                    mode=AnswerMode.INSUFFICIENT,
                    probs=(1.0, 0.0, 0.0),
                    reason="Pyrrho: no contexts retrieved.",
                )
                continue
            non_empty_positions.append(index)
            non_empty_contexts.append(contexts)

        if non_empty_contexts:
            self._load()
            context_texts = [
                [_format_evidence_item(context) for context in contexts]
                for contexts in non_empty_contexts
            ]
            for index, decision in zip(
                non_empty_positions,
                self._predict_context_batches(query, context_texts),
                strict=True,
            ):
                decisions[index] = decision

        return [decision for decision in decisions if decision is not None]

    def _load(self) -> None:
        """Load the tokenizer and ONNX model once per process."""
        if self._tokenizer is not None and self._model is not None:
            return

        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return

            import os

            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

            from huggingface_hub import snapshot_download

            model_dir = _resolve_model_dir(self._model_id)
            if model_dir is None:
                model_dir = _model_cache_dir(self._model_id)
                model_dir.mkdir(parents=True, exist_ok=True)
                snapshot_kwargs: dict[str, Any] = {
                    "repo_id": self._model_id,
                    "local_dir": model_dir,
                }
                allow_patterns = _snapshot_allow_patterns(self._model_id)
                if allow_patterns is not None:
                    snapshot_kwargs["allow_patterns"] = allow_patterns
                snapshot_download(**snapshot_kwargs)

            if not _is_v2_model_dir(model_dir):
                raise ValueError(
                    "Pyrrho model directory must be a v2 package with native "
                    f"evidence heads: {model_dir}"
                )

            onnx_path = _preferred_onnx_path(model_dir)
            if onnx_path is None:
                raise ValueError(
                    "Pyrrho v2 packages must include model.onnx or "
                    f"model_quantized.onnx: {model_dir}"
                )

            self._model_dir = model_dir
            self._tokenizer = _load_tokenizer(model_dir)
            self._model = _load_onnx_session(onnx_path)
            self._sufficient_threshold = _load_sufficient_threshold(model_dir) or TAU

    def _predict_context_batches(
        self,
        query: str,
        contexts_by_prefix: list[list[str]],
    ) -> list[GovernanceDecision]:
        """Run all non-empty evidence prefixes in one ONNX batch."""
        logits = self._run_onnx_texts(
            [_format_input(query, contexts) for contexts in contexts_by_prefix]
        )
        return [
            _v2_decision_from_logits(row, sufficient_threshold=self._sufficient_threshold)
            for row in logits
        ]

    def _run_onnx_texts(self, texts: list[str]) -> Any:
        """Tokenize and run a v2 classifier batch through ONNX Runtime."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Pyrrho was not loaded before inference.")

        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=V2_MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )
        declared_inputs = [node.name for node in self._model.get_inputs()]
        feed = {name: encoded[name] for name in declared_inputs if name in encoded}
        if not feed:
            raise RuntimeError("Pyrrho ONNX graph did not declare tokenizer-compatible inputs.")
        return self._model.run(None, feed)[0]


def _snapshot_allow_patterns(model_id: str) -> tuple[str, ...] | None:
    """Limit the default download to files needed by the ONNX runtime."""
    if model_id == MODEL_ID:
        return _DEFAULT_ALLOW_PATTERNS
    return None


def _preferred_onnx_path(model_dir: Path) -> Path | None:
    """Return the preferred packaged ONNX graph for a Pyrrho v2 package."""
    for filename in ONNX_MODEL_FILENAMES:
        path = model_dir / filename
        if path.exists() and path.is_file():
            return path
    return None


def _load_onnx_session(onnx_path: Path) -> Any:
    """Load a Pyrrho ONNX graph with the CPU execution provider."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Pyrrho v2 ONNX inference requires onnxruntime.") from exc
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _load_hf_config(model_dir: Path) -> dict[str, Any]:
    """Read a standard Hugging Face model config."""
    with (model_dir / "config.json").open(encoding="utf-8") as f:
        return json.load(f)


def _is_v2_model_dir(model_dir: Path) -> bool:
    """Return whether a directory is a v2 native-head classifier package."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        config = _load_hf_config(model_dir)
    except (OSError, json.JSONDecodeError):
        return False
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        return False
    labels = {str(value) for value in id2label.values()}
    return {
        "evidence_verdict.INSUFFICIENT",
        "evidence_verdict.DISPUTED",
        "evidence_verdict.SUFFICIENT",
        "failure_mode.none",
        "retrieval_intents.needs_lookup",
        "evidence_kinds.needs_text",
    }.issubset(labels)


def _model_cache_dir(model_id: str) -> Path:
    """Return Fitz's managed local cache directory for a Pyrrho model."""
    safe_id = model_id.replace("/", "__")
    return FitzPaths.user_home() / "models" / "pyrrho" / safe_id


def _resolve_model_dir(model_id: str) -> Path | None:
    """Return a local Pyrrho package directory when the model spec names one."""
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Pyrrho model path must be a directory: {candidate}")
        return candidate
    return None


def _load_sufficient_threshold(model_dir: Path) -> float | None:
    """Read the packaged release threshold when a Pyrrho manifest provides one."""
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("release", {}).get("sufficient_threshold")
        return float(value) if value is not None else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_tokenizer(model_dir: Path) -> Any:
    """Load the tokenizer without trusting repo-specific tokenizer metadata."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        model_max_length=8192,
    )


def _format_evidence_item(item: EvidenceItem) -> str:
    """Return the evidence text Pyrrho should govern over."""
    return str(getattr(item, "content", "") or getattr(item, "excerpt", "") or "")


def _v2_decision_from_logits(
    logits: Any,
    *,
    sufficient_threshold: float | None = TAU,
) -> GovernanceDecision:
    """Convert one v2 output row into the public decision object."""
    evidence_verdict, failure_mode, retrieval_intents, evidence_kinds = _v2_core_heads(
        logits,
        sufficient_threshold=sufficient_threshold,
    )
    probs = (
        float(evidence_verdict.probabilities["INSUFFICIENT"]),
        float(evidence_verdict.probabilities["DISPUTED"]),
        float(evidence_verdict.probabilities["SUFFICIENT"]),
    )
    mode = _VERDICT_TO_MODE[evidence_verdict.final_label]
    heads: dict[str, HeadDecision | MultiLabelDecision] = {
        "evidence_verdict": evidence_verdict,
        "failure_mode": failure_mode,
        "retrieval_intents": retrieval_intents,
        "evidence_kinds": evidence_kinds,
    }
    return GovernanceDecision(
        mode=mode,
        probs=probs,
        reason=_reason_for(mode, probs),
        evidence_verdict=evidence_verdict,
        failure_mode=failure_mode,
        retrieval_intents=retrieval_intents,
        evidence_kinds=evidence_kinds,
        heads=heads,
    )


def _v2_core_heads(
    logits: Any,
    *,
    sufficient_threshold: float | None,
) -> tuple[HeadDecision, HeadDecision, MultiLabelDecision, MultiLabelDecision]:
    """Decode the four native v2 output groups."""
    evidence_verdict = _single_label_decision(
        logits,
        0,
        _VERDICT_LABELS,
        sufficient_threshold=sufficient_threshold,
    )
    failure_mode = _single_label_decision(logits, 3, _FAILURE_LABELS)
    retrieval_intents = _multi_label_decision(logits, 8, _RETRIEVAL_INTENTS)
    evidence_kinds = _multi_label_decision(logits, 12, _EVIDENCE_KINDS)
    return evidence_verdict, failure_mode, retrieval_intents, evidence_kinds


def _single_label_decision(
    logits: Any,
    start: int,
    labels: tuple[str, ...],
    *,
    sufficient_threshold: float | None = None,
) -> HeadDecision:
    """Decode a mutually exclusive v2 label group."""
    probabilities = {
        label: probability
        for label, probability in zip(
            labels,
            _softmax(_slice_logits(logits, start, len(labels))),
            strict=True,
        )
    }
    final_label: str | None = None
    used_threshold_fallback = False
    raw_label = max(probabilities, key=probabilities.get)
    if (
        sufficient_threshold is not None
        and "SUFFICIENT" in probabilities
        and raw_label == "SUFFICIENT"
        and probabilities[raw_label] < sufficient_threshold
    ):
        fallback_labels = [label for label in labels if label != "SUFFICIENT"]
        final_label = max(fallback_labels, key=probabilities.get)
        used_threshold_fallback = True
    return _head_from_probabilities(
        probabilities,
        final_label=final_label,
        raw_label=raw_label,
        threshold=sufficient_threshold if "SUFFICIENT" in probabilities else None,
        used_threshold_fallback=used_threshold_fallback,
    )


def _multi_label_decision(
    logits: Any,
    start: int,
    labels: tuple[str, ...],
    *,
    threshold: float = 0.5,
) -> MultiLabelDecision:
    """Decode a multi-label v2 output group."""
    probabilities = {
        label: _sigmoid(value)
        for label, value in zip(labels, _slice_logits(logits, start, len(labels)), strict=True)
    }
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    raw_label = ranked[0]
    final_labels = tuple(label for label in labels if probabilities[label] >= threshold)
    if not final_labels:
        final_labels = (raw_label,)
    final_label = max(final_labels, key=lambda label: probabilities[label])
    runner_up_label = next((label for label in ranked if label != final_label), final_label)
    confidence = float(probabilities[final_label])
    runner_up_probability = float(probabilities[runner_up_label])
    return MultiLabelDecision(
        raw_label=raw_label,
        final_label=final_label,
        final_labels=final_labels,
        used_threshold_fallback=raw_label not in final_labels,
        threshold=threshold,
        confidence=confidence,
        probabilities={label: float(probability) for label, probability in probabilities.items()},
        runner_up_label=runner_up_label,
        runner_up_probability=runner_up_probability,
        margin_to_runner_up=float(confidence - runner_up_probability),
        entropy=_binary_entropy(probabilities.values()),
    )


def _head_from_probabilities(
    probabilities: dict[str, float],
    *,
    final_label: str | None = None,
    raw_label: str | None = None,
    threshold: float | None = None,
    used_threshold_fallback: bool = False,
) -> HeadDecision:
    """Build HeadDecision metadata from an already-normalized probability map."""
    if not probabilities:
        raise ValueError("Cannot build a head decision from empty probabilities.")
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    raw = raw_label or ranked[0]
    final = final_label or raw
    runner_up = next((label for label in ranked if label != final), final)
    confidence = float(probabilities[final])
    runner_up_probability = float(probabilities[runner_up])
    return HeadDecision(
        raw_label=raw,
        final_label=final,
        used_threshold_fallback=used_threshold_fallback,
        threshold=threshold,
        confidence=confidence,
        probabilities={label: float(value) for label, value in probabilities.items()},
        runner_up_label=runner_up,
        runner_up_probability=runner_up_probability,
        margin_to_runner_up=float(confidence - runner_up_probability),
        entropy=float(-sum(prob * math.log(prob) for prob in probabilities.values() if prob > 0.0)),
    )


def _slice_logits(logits: Any, start: int, size: int) -> list[float]:
    """Return a flat slice from one output row."""
    values = _to_float_list(logits)
    return values[start : start + size]


def _to_float_list(values: Any) -> list[float]:
    """Normalize one tensor/array/list row to a flat float list."""
    if hasattr(values, "detach"):
        raw = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        raw = values.tolist()
    else:
        raw = list(values)
    if not isinstance(raw, list):
        raw = [raw]
    return [float(value) for value in raw]


def _sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""
    if value >= 0:
        z = math.exp(-float(value))
        return 1.0 / (1.0 + z)
    z = math.exp(float(value))
    return z / (1.0 + z)


def _binary_entropy(probabilities: Iterable[float]) -> float:
    """Return binary entropy summed over independent labels."""
    entropy = 0.0
    for probability in probabilities:
        probability = max(0.0, min(1.0, float(probability)))
        if probability > 0.0:
            entropy -= probability * math.log(probability)
        inverse = 1.0 - probability
        if inverse > 0.0:
            entropy -= inverse * math.log(inverse)
    return float(entropy)


def _softmax(logits: list[float]) -> list[float]:
    """Softmax one logits row with numerical stabilization."""
    offset = max(logits)
    exp = [math.exp(float(value) - offset) for value in logits]
    total = sum(exp)
    return [value / total for value in exp]


__all__ = [
    "GovernanceDecision",
    "HeadDecision",
    "LABELS",
    "MODEL_ID",
    "MultiLabelDecision",
    "PYRRHO_POST_TAG",
    "PYRRHO_PRE_TAG",
    "Pyrrho",
    "PyrrhoQueryPlan",
    "TAU",
]
