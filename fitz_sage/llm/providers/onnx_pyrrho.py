"""Pinned ONNX adapter and mechanical decoder for the Pyrrho model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from fitz_sage.llm.providers.pyrrho_schema import (
    EVIDENCE_KIND_KEYS,
    EVIDENCE_VERDICTS,
    FAILURE_MODES,
    NUM_PYRRHO_LABELS,
    RETRIEVAL_INTENT_KEYS,
    build_pyrrho_evidence_text,
    build_pyrrho_query_text,
    pyrrho_label_names,
)
from fitz_sage.llm.providers.pyrrho_types import (
    GovernanceDecision,
    HeadDecision,
    MultiLabelDecision,
    PyrrhoModelIdentity,
    PyrrhoQueryPlan,
)

DEFAULT_MODEL_ID = "yafitzdev/pyrrho-v2-nano-g1"
DEFAULT_MODEL_REVISION = "948f0500b74871cfaec7689a01d4eab0dd516e1b"
DEFAULT_MAX_INPUT_TOKENS = 2048
DEFAULT_SUFFICIENT_THRESHOLD = 0.34
ONNX_MODEL_FILENAMES = ("model.onnx", "model_quantized.onnx")
DEFAULT_ONNX_MODEL_FILENAME = "model.onnx"

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
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_VALID_FAILURES_BY_VERDICT = {
    "SUFFICIENT": frozenset({"none"}),
    "DISPUTED": frozenset({"unresolved_conflict"}),
    "INSUFFICIENT": frozenset(
        {
            "missing_or_incomplete_evidence",
            "wrong_scope_or_version",
            "ambiguous_request",
        }
    ),
}


@dataclass(frozen=True)
class _InferenceInputStats:
    input_tokens: int
    input_truncated: bool
    max_input_tokens: int


class OnnxPyrrho:
    """Managed ONNX adapter for the Pyrrho governance model.

    The adapter implements the model artifact's input, output, and decoding
    contract. Retrieval code consumes the resulting verdict without adding a
    second governance policy.
    """

    def __init__(self, model_spec: str | os.PathLike[str] = DEFAULT_MODEL_ID) -> None:
        self._model_spec = os.fspath(model_spec)
        self._lock = threading.Lock()
        self._model_dir: Path | None = None
        self._onnx_path: Path | None = None
        self._tokenizer: Any = None
        self._session: Any = None
        self._sufficient_threshold = DEFAULT_SUFFICIENT_THRESHOLD
        self._max_input_tokens = DEFAULT_MAX_INPUT_TOKENS
        self._identity: PyrrhoModelIdentity | None = None

    @property
    def model_spec(self) -> str:
        return self._model_spec

    @property
    def identity(self) -> PyrrhoModelIdentity:
        """Return the exact loaded model identity."""
        self._load()
        if self._identity is None:
            raise RuntimeError("Pyrrho loaded without establishing model identity.")
        return self._identity

    def plan_query(self, query: str) -> PyrrhoQueryPlan:
        """Classify query shape before retrieval using Pyrrho's native heads."""
        self._load()
        logits, input_stats = self._run_texts([build_pyrrho_query_text(query)])
        _, _, retrieval_intents, evidence_kinds = _decode_core_heads(
            logits[0],
            sufficient_threshold=None,
        )
        stats = input_stats[0]
        return PyrrhoQueryPlan(
            retrieval_intents=retrieval_intents,
            evidence_kinds=evidence_kinds,
            input_tokens=stats.input_tokens,
            input_truncated=stats.input_truncated,
            max_input_tokens=stats.max_input_tokens,
        )

    def decide(self, query: str, evidence: Sequence[Any]) -> GovernanceDecision:
        """Return Pyrrho's final verdict for one complete evidence set."""
        normalized = normalize_evidence(evidence)
        if not normalized:
            return empty_evidence_decision()

        self._load()
        logits, input_stats = self._run_texts([build_pyrrho_evidence_text(query, normalized)])
        model = self.identity.to_dict()
        return decision_from_logits(
            logits[0],
            sufficient_threshold=self._sufficient_threshold,
            input_tokens=input_stats[0].input_tokens,
            input_truncated=input_stats[0].input_truncated,
            max_input_tokens=input_stats[0].max_input_tokens,
            model=model,
        )

    def decide_many(
        self,
        query: str,
        evidence_sets: Sequence[Sequence[Any]],
    ) -> list[GovernanceDecision]:
        """Classify independent complete evidence sets in one ONNX batch."""
        normalized_sets = [normalize_evidence(evidence) for evidence in evidence_sets]
        decisions: list[GovernanceDecision | None] = [None] * len(normalized_sets)
        model_positions = [index for index, evidence in enumerate(normalized_sets) if evidence]
        if not model_positions:
            return [empty_evidence_decision() for _ in normalized_sets]

        self._load()
        texts = [
            build_pyrrho_evidence_text(query, normalized_sets[index]) for index in model_positions
        ]
        logits, input_stats = self._run_texts(texts)
        model = self.identity.to_dict()
        for output_index, evidence_index in enumerate(model_positions):
            stats = input_stats[output_index]
            decisions[evidence_index] = decision_from_logits(
                logits[output_index],
                sufficient_threshold=self._sufficient_threshold,
                input_tokens=stats.input_tokens,
                input_truncated=stats.input_truncated,
                max_input_tokens=stats.max_input_tokens,
                model=model,
            )
        for index, evidence in enumerate(normalized_sets):
            if not evidence:
                decisions[index] = empty_evidence_decision()
        return [cast(GovernanceDecision, decision) for decision in decisions]

    def _load(self) -> None:
        if self._tokenizer is not None and self._session is not None:
            return
        with self._lock:
            if self._tokenizer is not None and self._session is not None:
                return

            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

            model_dir = _resolve_model_dir(self._model_spec)
            if model_dir is None:
                from huggingface_hub import snapshot_download

                repo_id, revision = _pinned_remote_model(self._model_spec)
                download_options: dict[str, Any] = {
                    "repo_id": repo_id,
                    "revision": revision,
                }
                if repo_id == DEFAULT_MODEL_ID:
                    download_options["allow_patterns"] = _DEFAULT_ALLOW_PATTERNS
                model_dir = Path(snapshot_download(**download_options))

            artifact = _validate_model_artifact(model_dir)
            tokenizer = _load_tokenizer(model_dir)
            session = _load_onnx_session(artifact.onnx_path)

            self._model_dir = model_dir
            self._onnx_path = artifact.onnx_path
            self._tokenizer = tokenizer
            self._session = session
            self._sufficient_threshold = artifact.sufficient_threshold
            self._max_input_tokens = artifact.max_input_tokens
            self._identity = PyrrhoModelIdentity(
                model_spec=self._model_spec,
                model_directory=str(model_dir.resolve()),
                graph=artifact.onnx_path.name,
                graph_sha256=_sha256_file(artifact.onnx_path),
                max_input_tokens=artifact.max_input_tokens,
                sufficient_threshold=artifact.sufficient_threshold,
            )

    def _run_texts(self, texts: list[str]) -> tuple[Any, list[_InferenceInputStats]]:
        if self._tokenizer is None or self._session is None:
            raise RuntimeError("Pyrrho was not loaded before inference.")

        untruncated = self._tokenizer(
            texts,
            truncation=False,
            padding=False,
            verbose=False,
        )
        raw_input_ids = untruncated.get("input_ids")
        if not isinstance(raw_input_ids, list) or len(raw_input_ids) != len(texts):
            raise RuntimeError("Pyrrho tokenizer did not return one token sequence per input.")
        token_lengths = [len(input_ids) for input_ids in raw_input_ids]

        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=self._max_input_tokens,
            padding=True,
            return_tensors="np",
        )
        declared_inputs = [node.name for node in self._session.get_inputs()]
        feed = {name: encoded[name] for name in declared_inputs if name in encoded}
        if set(feed) != set(declared_inputs):
            missing = sorted(set(declared_inputs) - set(feed))
            raise RuntimeError(f"Pyrrho tokenizer cannot feed ONNX graph inputs: {missing}.")

        outputs = self._session.run(None, feed)
        if not outputs:
            raise RuntimeError("Pyrrho ONNX graph returned no outputs.")
        logits = outputs[0]
        shape = getattr(logits, "shape", None)
        if shape is None or tuple(shape) != (len(texts), NUM_PYRRHO_LABELS):
            raise RuntimeError(
                "Pyrrho ONNX graph must return logits with shape "
                f"(batch, {NUM_PYRRHO_LABELS}); got {shape!r}."
            )
        stats = [
            _InferenceInputStats(
                input_tokens=length,
                input_truncated=length > self._max_input_tokens,
                max_input_tokens=self._max_input_tokens,
            )
            for length in token_lengths
        ]
        return logits, stats


@dataclass(frozen=True)
class _ValidatedModel:
    onnx_path: Path
    sufficient_threshold: float
    max_input_tokens: int


def normalize_evidence(evidence: Iterable[Any]) -> list[dict[str, str]]:
    """Normalize common evidence objects without changing their text."""
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            source_id = str(index)
            text = item
        elif isinstance(item, Mapping):
            source_id = str(
                item.get("source_id") or item.get("document_id") or item.get("id") or index
            )
            text = str(item.get("text") or item.get("content") or item.get("excerpt") or "")
        else:
            source_id = str(
                getattr(item, "source_id", None)
                or getattr(item, "document_id", None)
                or getattr(item, "id", None)
                or index
            )
            text = str(
                getattr(item, "text", None)
                or getattr(item, "content", None)
                or getattr(item, "excerpt", None)
                or ""
            )
        normalized.append({"source_id": source_id, "text": text})
    return normalized


def empty_evidence_decision() -> GovernanceDecision:
    """Return Pyrrho's deterministic verdict when no evidence exists."""
    verdict_probabilities = {
        "INSUFFICIENT": 1.0,
        "DISPUTED": 0.0,
        "SUFFICIENT": 0.0,
    }
    failure_probabilities = {
        "none": 0.0,
        "unresolved_conflict": 0.0,
        "missing_or_incomplete_evidence": 1.0,
        "wrong_scope_or_version": 0.0,
        "ambiguous_request": 0.0,
    }
    evidence_verdict = _head_from_probabilities(
        verdict_probabilities,
        final_label="INSUFFICIENT",
        raw_label="INSUFFICIENT",
        threshold=DEFAULT_SUFFICIENT_THRESHOLD,
    )
    failure_mode = _head_from_probabilities(
        failure_probabilities,
        final_label="missing_or_incomplete_evidence",
        raw_label="missing_or_incomplete_evidence",
    )
    return GovernanceDecision(
        verdict="INSUFFICIENT",
        reason="Pyrrho: no evidence was provided.",
        probabilities=verdict_probabilities,
        evidence_verdict=evidence_verdict,
        failure_mode=failure_mode,
        deterministic=True,
    )


def decision_from_logits(
    logits: Any,
    *,
    sufficient_threshold: float = DEFAULT_SUFFICIENT_THRESHOLD,
    input_tokens: int = 0,
    input_truncated: bool = False,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    model: Mapping[str, Any] | None = None,
) -> GovernanceDecision:
    """Decode one v2 output row using Pyrrho's canonical runtime policy."""
    evidence_verdict, failure_mode, retrieval_intents, evidence_kinds = _decode_core_heads(
        logits,
        sufficient_threshold=sufficient_threshold,
    )
    evidence_verdict, failure_mode, consistency_reason, original_pair = _reconcile_verdict_failure(
        evidence_verdict, failure_mode
    )
    probabilities = dict(evidence_verdict.probabilities)
    verdict = evidence_verdict.final_label
    reason = _reason_for(verdict, probabilities)
    if consistency_reason is not None:
        reason = f"{reason} {consistency_reason}"
    return GovernanceDecision(
        verdict=verdict,
        reason=reason,
        probabilities=probabilities,
        evidence_verdict=evidence_verdict,
        failure_mode=failure_mode,
        retrieval_intents=retrieval_intents,
        evidence_kinds=evidence_kinds,
        input_tokens=input_tokens,
        input_truncated=input_truncated,
        max_input_tokens=max_input_tokens,
        consistency_applied=consistency_reason is not None,
        consistency_reason=consistency_reason,
        pre_consistency_pair=original_pair,
        model=dict(model or {}),
    )


def query_plan_from_logits(
    logits: Any,
    *,
    input_tokens: int = 0,
    input_truncated: bool = False,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> PyrrhoQueryPlan:
    """Decode the query-only heads with the same semantics as the runtime."""
    _, _, retrieval_intents, evidence_kinds = _decode_core_heads(
        logits,
        sufficient_threshold=None,
    )
    return PyrrhoQueryPlan(
        retrieval_intents=retrieval_intents,
        evidence_kinds=evidence_kinds,
        input_tokens=input_tokens,
        input_truncated=input_truncated,
        max_input_tokens=max_input_tokens,
    )


def _reason_for(verdict: str, probabilities: Mapping[str, float]) -> str:
    probability = probabilities[verdict]
    if verdict == "SUFFICIENT":
        return f"Pyrrho: evidence is sufficient for a confident answer (P={probability:.2f})."
    if verdict == "DISPUTED":
        return f"Pyrrho: evidence is disputed (P={probability:.2f})."
    return f"Pyrrho: evidence is insufficient (P={probability:.2f})."


def _decode_core_heads(
    logits: Any,
    *,
    sufficient_threshold: float | None,
) -> tuple[HeadDecision, HeadDecision, MultiLabelDecision, MultiLabelDecision]:
    values = _to_float_list(logits)
    if len(values) != NUM_PYRRHO_LABELS:
        raise ValueError(f"Expected {NUM_PYRRHO_LABELS} Pyrrho logits, got {len(values)}.")
    evidence_verdict = _single_label_decision(
        values,
        0,
        EVIDENCE_VERDICTS,
        sufficient_threshold=sufficient_threshold,
    )
    failure_mode = _single_label_decision(values, 3, FAILURE_MODES)
    retrieval_intents = _multi_label_decision(values, 8, RETRIEVAL_INTENT_KEYS)
    evidence_kinds = _multi_label_decision(values, 12, EVIDENCE_KIND_KEYS)
    return evidence_verdict, failure_mode, retrieval_intents, evidence_kinds


def _single_label_decision(
    logits: list[float],
    start: int,
    labels: tuple[str, ...],
    *,
    sufficient_threshold: float | None = None,
) -> HeadDecision:
    probabilities = dict(zip(labels, _softmax(logits[start : start + len(labels)]), strict=True))
    raw_label = max(probabilities, key=probabilities.__getitem__)
    final_label = raw_label
    threshold_applied = False
    if (
        sufficient_threshold is not None
        and raw_label == "SUFFICIENT"
        and probabilities[raw_label] < sufficient_threshold
    ):
        final_label = max(
            (label for label in labels if label != "SUFFICIENT"),
            key=probabilities.__getitem__,
        )
        threshold_applied = True
    return _head_from_probabilities(
        probabilities,
        final_label=final_label,
        raw_label=raw_label,
        threshold=sufficient_threshold if "SUFFICIENT" in probabilities else None,
        threshold_applied=threshold_applied,
    )


def _multi_label_decision(
    logits: list[float],
    start: int,
    labels: tuple[str, ...],
    *,
    threshold: float = 0.5,
) -> MultiLabelDecision:
    probabilities = {
        label: _sigmoid(value)
        for label, value in zip(
            labels,
            logits[start : start + len(labels)],
            strict=True,
        )
    }
    ranked = sorted(probabilities, key=probabilities.__getitem__, reverse=True)
    raw_label = ranked[0]
    final_labels = tuple(label for label in labels if probabilities[label] >= threshold)
    threshold_applied = False
    if not final_labels:
        final_labels = (raw_label,)
        threshold_applied = True
    final_label = max(final_labels, key=probabilities.__getitem__)
    runner_up_label = next(
        (label for label in ranked if label != final_label),
        final_label,
    )
    confidence = float(probabilities[final_label])
    runner_up_probability = float(probabilities[runner_up_label])
    return MultiLabelDecision(
        raw_label=raw_label,
        final_label=final_label,
        final_labels=final_labels,
        probabilities={label: float(value) for label, value in probabilities.items()},
        confidence=confidence,
        runner_up_label=runner_up_label,
        runner_up_probability=runner_up_probability,
        margin_to_runner_up=confidence - runner_up_probability,
        entropy=_binary_entropy(probabilities.values()),
        threshold=threshold,
        threshold_applied=threshold_applied,
    )


def _reconcile_verdict_failure(
    evidence_verdict: HeadDecision,
    failure_mode: HeadDecision,
) -> tuple[HeadDecision, HeadDecision, str | None, tuple[str, str]]:
    original_pair = (evidence_verdict.final_label, failure_mode.final_label)
    if failure_mode.final_label in _VALID_FAILURES_BY_VERDICT[evidence_verdict.final_label]:
        return evidence_verdict, failure_mode, None, original_pair

    final_verdict = evidence_verdict.final_label
    final_failure = failure_mode.final_label
    if final_failure == "unresolved_conflict" and final_verdict != "INSUFFICIENT":
        final_verdict = "DISPUTED"
    elif final_failure in _VALID_FAILURES_BY_VERDICT["INSUFFICIENT"]:
        final_verdict = "INSUFFICIENT"
    else:
        final_verdict = "INSUFFICIENT"
        final_failure = "missing_or_incomplete_evidence"

    if (
        final_verdict == "INSUFFICIENT"
        and final_failure not in _VALID_FAILURES_BY_VERDICT["INSUFFICIENT"]
    ):
        final_failure = "missing_or_incomplete_evidence"

    final_pair = (final_verdict, final_failure)
    reason = (
        "Pyrrho consistency policy changed contradictory heads "
        f"{original_pair[0]}+{original_pair[1]} to {final_pair[0]}+{final_pair[1]}."
    )
    if final_verdict != evidence_verdict.final_label:
        evidence_verdict = _head_with_consistent_final(
            evidence_verdict,
            final_verdict,
            reason,
        )
    if final_failure != failure_mode.final_label:
        failure_mode = _head_with_consistent_final(
            failure_mode,
            final_failure,
            reason,
        )
    if failure_mode.final_label not in _VALID_FAILURES_BY_VERDICT[evidence_verdict.final_label]:
        raise AssertionError("Pyrrho consistency policy produced an invalid head pair.")
    return evidence_verdict, failure_mode, reason, original_pair


def _head_with_consistent_final(
    head: HeadDecision,
    final_label: str,
    reason: str,
) -> HeadDecision:
    rebuilt = _head_from_probabilities(
        head.probabilities,
        final_label=final_label,
        raw_label=head.raw_label,
        threshold=head.threshold,
        threshold_applied=head.threshold_applied,
    )
    return replace(
        rebuilt,
        consistency_applied=True,
        consistency_reason=reason,
    )


def _head_from_probabilities(
    probabilities: Mapping[str, float],
    *,
    final_label: str,
    raw_label: str,
    threshold: float | None = None,
    threshold_applied: bool = False,
) -> HeadDecision:
    if not probabilities:
        raise ValueError("Cannot build a head decision from empty probabilities.")
    ranked = sorted(probabilities, key=probabilities.__getitem__, reverse=True)
    runner_up_label = next(
        (label for label in ranked if label != final_label),
        final_label,
    )
    confidence = float(probabilities[final_label])
    runner_up_probability = float(probabilities[runner_up_label])
    return HeadDecision(
        raw_label=raw_label,
        final_label=final_label,
        probabilities={label: float(value) for label, value in probabilities.items()},
        confidence=confidence,
        runner_up_label=runner_up_label,
        runner_up_probability=runner_up_probability,
        margin_to_runner_up=confidence - runner_up_probability,
        entropy=float(
            -sum(
                probability * math.log(probability)
                for probability in probabilities.values()
                if probability > 0.0
            )
        ),
        threshold=threshold,
        threshold_applied=threshold_applied,
    )


def _to_float_list(values: Any) -> list[float]:
    if hasattr(values, "detach"):
        raw = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        raw = values.tolist()
    else:
        raw = list(values)
    if not isinstance(raw, list):
        raw = [raw]
    return [float(value) for value in raw]


def _softmax(logits: list[float]) -> list[float]:
    offset = max(logits)
    exponentials = [math.exp(value - offset) for value in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_entropy(probabilities: Iterable[float]) -> float:
    entropy = 0.0
    for probability in probabilities:
        bounded = max(0.0, min(1.0, float(probability)))
        if bounded > 0.0:
            entropy -= bounded * math.log(bounded)
        inverse = 1.0 - bounded
        if inverse > 0.0:
            entropy -= inverse * math.log(inverse)
    return entropy


def _resolve_model_dir(model_spec: str) -> Path | None:
    candidate = Path(model_spec).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Pyrrho model path must be a directory: {candidate}")
        return candidate
    return None


def _pinned_remote_model(model_spec: str) -> tuple[str, str]:
    if model_spec == DEFAULT_MODEL_ID:
        return DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION

    if "@" not in model_spec:
        raise ValueError(
            "Remote Pyrrho models must pin an immutable Hub commit as "
            "owner/repo@40-character-commit."
        )
    repo_id, revision = model_spec.rsplit("@", 1)
    if not repo_id or not _PINNED_REVISION.fullmatch(revision):
        raise ValueError(
            "Remote Pyrrho model revision must be a 40-character hexadecimal Hub commit."
        )
    return repo_id, revision.lower()


def _validate_model_artifact(model_dir: Path) -> _ValidatedModel:
    config = _load_hf_config(model_dir)
    expected_names = tuple(pyrrho_label_names())
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        raise ValueError("Pyrrho config.id2label must be an object.")
    labels = tuple(
        str(id2label.get(str(index), id2label.get(index, ""))) for index in range(NUM_PYRRHO_LABELS)
    )
    expected_label2id = {label: index for index, label in enumerate(expected_names)}
    if (
        len(id2label) != NUM_PYRRHO_LABELS
        or labels != expected_names
        or config.get("label2id") != expected_label2id
        or config.get("problem_type") != "multi_label_classification"
    ):
        raise ValueError(
            "Pyrrho model artifact must expose the exact v2 18-label native-head schema."
        )
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise ValueError(f"Pyrrho model artifact is missing tokenizer.json: {model_dir}")

    manifest = _load_release_manifest(model_dir)
    onnx_path = _preferred_onnx_path(model_dir, manifest)
    threshold = _load_sufficient_threshold(manifest)
    max_input_tokens = _load_max_input_tokens(config, manifest)
    return _ValidatedModel(
        onnx_path=onnx_path,
        sufficient_threshold=threshold,
        max_input_tokens=max_input_tokens,
    )


def _load_hf_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Pyrrho model config: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Pyrrho config must be a JSON object: {config_path}")
    return cast(dict[str, Any], loaded)


def _load_release_manifest(model_dir: Path) -> dict[str, Any] | None:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Pyrrho release manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Pyrrho release manifest must be an object: {manifest_path}")
    if "release" in manifest and not isinstance(manifest["release"], dict):
        raise ValueError(f"Pyrrho release metadata must be an object: {manifest_path}")
    return cast(dict[str, Any], manifest)


def _preferred_onnx_path(
    model_dir: Path,
    manifest: Mapping[str, Any] | None,
) -> Path:
    release = manifest.get("release") if manifest is not None else None
    declared = release.get("preferred_onnx_graph") if isinstance(release, Mapping) else None
    if declared is not None and declared not in ONNX_MODEL_FILENAMES:
        raise ValueError("release.preferred_onnx_graph must be model.onnx or model_quantized.onnx.")
    filename = str(declared or DEFAULT_ONNX_MODEL_FILENAME)
    onnx_path = model_dir / filename
    if not onnx_path.is_file():
        if declared is not None:
            raise ValueError(f"Manifest-declared Pyrrho ONNX graph is missing: {onnx_path}")
        raise ValueError(f"Pyrrho model artifact is missing model.onnx: {model_dir}")
    _verify_onnx_parity(model_dir, manifest, onnx_path)
    return onnx_path


def _verify_onnx_parity(
    model_dir: Path,
    manifest: Mapping[str, Any] | None,
    onnx_path: Path,
) -> None:
    parity = manifest.get("onnx_parity") if manifest is not None else None
    if parity is None:
        if onnx_path.name == "model_quantized.onnx":
            raise ValueError("INT8 Pyrrho selection requires a passed ONNX parity report.")
        return
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError("Pyrrho release ONNX parity is absent or not passed.")

    report_name = parity.get("report")
    report_hash = str(parity.get("report_sha256") or "")
    if (
        not isinstance(report_name, str)
        or not report_name
        or Path(report_name).name != report_name
        or not _SHA256.fullmatch(report_hash)
    ):
        raise ValueError("Pyrrho ONNX parity report declaration is invalid.")
    report_path = model_dir / report_name
    if not report_path.is_file() or _sha256_file(report_path) != report_hash.lower():
        raise ValueError("Pyrrho ONNX parity report hash mismatch.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Pyrrho ONNX parity report is invalid.") from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("Pyrrho ONNX parity report did not pass.")

    comparison_name = (
        "native_vs_int8" if onnx_path.name == "model_quantized.onnx" else "native_vs_fp32"
    )
    comparisons = report.get("comparisons")
    comparison = comparisons.get(comparison_name) if isinstance(comparisons, Mapping) else None
    if not isinstance(comparison, Mapping) or comparison.get("passed") is not True:
        raise ValueError(f"Pyrrho parity report lacks a passing {comparison_name} comparison.")
    artifacts = report.get("artifacts")
    artifact_key = (
        "onnx_int8_sha256" if onnx_path.name == "model_quantized.onnx" else "onnx_fp32_sha256"
    )
    expected_hash = artifacts.get(artifact_key) if isinstance(artifacts, Mapping) else None
    if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
        raise ValueError("Pyrrho parity report lacks a valid selected-graph hash.")
    if _sha256_file(onnx_path) != expected_hash.lower():
        raise ValueError("Pyrrho selected ONNX graph hash differs from its parity report.")


def _load_sufficient_threshold(manifest: Mapping[str, Any] | None) -> float:
    release = manifest.get("release") if manifest is not None else None
    value = release.get("sufficient_threshold") if isinstance(release, Mapping) else None
    if value is None:
        return DEFAULT_SUFFICIENT_THRESHOLD
    if isinstance(value, bool):
        raise ValueError("release.sufficient_threshold must be numeric.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("release.sufficient_threshold must be numeric.") from exc
    if not math.isfinite(threshold) or not (1.0 / 3.0) <= threshold <= 1.0:
        raise ValueError("release.sufficient_threshold must be finite and between 1/3 and 1.")
    return threshold


def _load_max_input_tokens(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> int:
    release = manifest.get("release") if manifest is not None else None
    value = release.get("max_length") if isinstance(release, Mapping) else None
    max_input_tokens = DEFAULT_MAX_INPUT_TOKENS if value is None else value
    if (
        isinstance(max_input_tokens, bool)
        or not isinstance(max_input_tokens, int)
        or max_input_tokens <= 0
    ):
        raise ValueError("release.max_length must be a positive integer.")
    encoder_limit = config.get("max_position_embeddings")
    if isinstance(encoder_limit, bool) or not isinstance(encoder_limit, int) or encoder_limit <= 0:
        raise ValueError("config.max_position_embeddings must be a positive integer.")
    if max_input_tokens > encoder_limit:
        raise ValueError(
            f"release.max_length={max_input_tokens} exceeds encoder limit {encoder_limit}."
        )
    return max_input_tokens


def _load_tokenizer(model_dir: Path) -> Any:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_dir / "tokenizer.json"),
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        model_max_length=8192,
    )
    tokenizer.truncation_side = "right"
    return tokenizer


def _load_onnx_session(onnx_path: Path) -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Pyrrho ONNX inference requires onnxruntime.") from exc
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.get_outputs()
    if not outputs:
        raise ValueError(f"Pyrrho ONNX graph has no outputs: {onnx_path}")
    output_shape = getattr(outputs[0], "shape", None)
    last_dimension = output_shape[-1] if output_shape else None
    if isinstance(last_dimension, int) and last_dimension != NUM_PYRRHO_LABELS:
        raise ValueError(
            f"Pyrrho ONNX graph must expose {NUM_PYRRHO_LABELS} logits; "
            f"got {last_dimension}: {onnx_path}"
        )
    return session


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "DEFAULT_SUFFICIENT_THRESHOLD",
    "OnnxPyrrho",
    "decision_from_logits",
    "empty_evidence_decision",
    "normalize_evidence",
    "query_plan_from_logits",
]
