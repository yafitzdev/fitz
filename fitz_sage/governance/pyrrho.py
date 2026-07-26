# fitz_sage/governance/pyrrho.py
"""Pyrrho v2 ONNX governance backend."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.core.paths import FitzPaths
from fitz_sage.governance.protocol import EvidenceItem

MODEL_ID = "yafitzdev/pyrrho-v2-nano-g1"
COMPROMISED_MODEL_REVISION = "948f0500b74871cfaec7689a01d4eab0dd516e1b"
COMPROMISED_MODEL_OPT_IN = "FITZ_ALLOW_COMPROMISED_PYRRHO"
V2_MAX_LENGTH = 4096
TAU = 0.34
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
_V2_LABEL_NAMES = (
    *(f"evidence_verdict.{label}" for label in _VERDICT_LABELS),
    *(f"failure_mode.{label}" for label in _FAILURE_LABELS),
    *(f"retrieval_intents.{label}" for label in _RETRIEVAL_INTENTS),
    *(f"evidence_kinds.{label}" for label in _EVIDENCE_KINDS),
)
_NUM_V2_LABELS = len(_V2_LABEL_NAMES)
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
    used_consistency_fallback: bool = False
    consistency_reason: str | None = None


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
    input_tokens: int | None = None
    input_truncated: bool = False
    max_input_tokens: int | None = None
    used_consistency_fallback: bool = False
    consistency_reason: str | None = None
    pre_consistency_pair: tuple[str, str] | None = None

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
    input_tokens: int | None = None
    input_truncated: bool = False
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class _InferenceInputStats:
    """Token-budget telemetry for one ONNX input."""

    input_tokens: int
    input_truncated: bool
    max_input_tokens: int


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
        self._onnx_path: Path | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._sufficient_threshold = TAU
        self._max_length = V2_MAX_LENGTH

    def decide(self, query: str, contexts: Sequence[EvidenceItem]) -> GovernanceDecision:
        """Classify one (query, contexts) pair into a governance mode."""
        return self.decide_many(query, [contexts])[0]

    def plan_query(self, query: str) -> PyrrhoQueryPlan:
        """Classify one query before retrieval into native v2 planning heads."""
        self._load()
        logits, input_stats = self._run_onnx_texts_with_stats([_format_query_input(query)])
        _, _, retrieval_intents, evidence_kinds = _v2_core_heads(
            logits[0],
            sufficient_threshold=None,
        )
        stats = input_stats[0]
        return PyrrhoQueryPlan(
            retrieval_intents=retrieval_intents,
            evidence_kinds=evidence_kinds,
            heads={
                "retrieval_intents": retrieval_intents,
                "evidence_kinds": evidence_kinds,
            },
            input_tokens=stats.input_tokens,
            input_truncated=stats.input_truncated,
            max_input_tokens=stats.max_input_tokens,
        )

    def decide_many(
        self,
        query: str,
        contexts_by_prefix: Sequence[Sequence[EvidenceItem]],
    ) -> list[GovernanceDecision]:
        """Classify several evidence prefixes in one model batch."""
        decisions: list[GovernanceDecision | None] = [None] * len(contexts_by_prefix)
        non_empty_contexts: list[Sequence[EvidenceItem]] = []
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

            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

            from huggingface_hub import snapshot_download

            model_dir = _resolve_model_dir(self._model_id)
            if model_dir is None:
                repo_id, revision = _pinned_remote_model(self._model_id)
                model_dir = _model_cache_dir(repo_id, revision=revision)
                model_dir.mkdir(parents=True, exist_ok=True)
                snapshot_kwargs: dict[str, Any] = {
                    "repo_id": repo_id,
                    "revision": revision,
                    "local_dir": model_dir,
                }
                allow_patterns = _snapshot_allow_patterns(repo_id)
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

            threshold = _load_sufficient_threshold(model_dir)
            max_length = _load_max_length(model_dir)
            tokenizer = _load_tokenizer(model_dir)
            model = _load_onnx_session(onnx_path)

            # Commit loaded state only after every package and runtime check passes.
            self._model_dir = model_dir
            self._onnx_path = onnx_path
            self._tokenizer = tokenizer
            self._model = model
            self._sufficient_threshold = TAU if threshold is None else threshold
            self._max_length = max_length

    def _predict_context_batches(
        self,
        query: str,
        contexts_by_prefix: list[list[str]],
    ) -> list[GovernanceDecision]:
        """Run all non-empty evidence prefixes in one ONNX batch."""
        logits, input_stats = self._run_onnx_texts_with_stats(
            [_format_input(query, contexts) for contexts in contexts_by_prefix]
        )
        return [
            _v2_decision_from_logits(
                row,
                sufficient_threshold=self._sufficient_threshold,
                input_stats=stats,
            )
            for row, stats in zip(logits, input_stats, strict=True)
        ]

    def _run_onnx_texts(self, texts: list[str]) -> Any:
        """Tokenize and run a v2 classifier batch through ONNX Runtime."""
        logits, _ = self._run_onnx_texts_with_stats(texts)
        return logits

    def _run_onnx_texts_with_stats(
        self, texts: list[str]
    ) -> tuple[Any, list[_InferenceInputStats]]:
        """Run ONNX and return exact token-budget telemetry for every input."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Pyrrho was not loaded before inference.")

        max_length = int(getattr(self, "_max_length", V2_MAX_LENGTH))
        untruncated = self._tokenizer(
            texts,
            truncation=False,
            padding=False,
        )
        raw_input_ids = untruncated.get("input_ids")
        if not isinstance(raw_input_ids, list) or len(raw_input_ids) != len(texts):
            raise RuntimeError("Pyrrho tokenizer did not return one token sequence per input.")
        token_lengths = [len(input_ids) for input_ids in raw_input_ids]
        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="np",
        )
        declared_inputs = [node.name for node in self._model.get_inputs()]
        feed = {name: encoded[name] for name in declared_inputs if name in encoded}
        if set(feed) != set(declared_inputs):
            missing = sorted(set(declared_inputs) - set(feed))
            raise RuntimeError(f"Pyrrho tokenizer cannot feed ONNX graph inputs: {missing}.")
        outputs = self._model.run(None, feed)
        if not outputs:
            raise RuntimeError("Pyrrho ONNX graph returned no outputs.")
        logits = outputs[0]
        shape = getattr(logits, "shape", None)
        if shape is None or tuple(shape) != (len(texts), _NUM_V2_LABELS):
            raise RuntimeError(
                "Pyrrho ONNX graph must return logits with shape "
                f"(batch, {_NUM_V2_LABELS}); got {shape!r}."
            )
        stats = [
            _InferenceInputStats(
                input_tokens=length,
                input_truncated=length > max_length,
                max_input_tokens=max_length,
            )
            for length in token_lengths
        ]
        return logits, stats


def _snapshot_allow_patterns(model_id: str) -> tuple[str, ...] | None:
    """Limit the default download to files needed by the ONNX runtime."""
    if model_id == MODEL_ID:
        return _DEFAULT_ALLOW_PATTERNS
    return None


def _pinned_remote_model(model_id: str) -> tuple[str, str]:
    """Resolve a remote model to an immutable Hub revision or fail closed."""
    if model_id == MODEL_ID:
        opted_in = os.environ.get(COMPROMISED_MODEL_OPT_IN, "").strip().lower()
        if opted_in not in {"1", "true", "yes"}:
            raise RuntimeError(
                "The default remote Pyrrho model is quarantined because its training "
                "corpus contained benchmark-derived deterministic rows. Supply an "
                "explicit local clean model directory, or set "
                f"{COMPROMISED_MODEL_OPT_IN}=1 only to reproduce the compromised "
                "historical artifact for forensic work."
            )
        return MODEL_ID, COMPROMISED_MODEL_REVISION

    if "@" not in model_id:
        raise ValueError(
            "Remote Pyrrho models must pin an immutable 40-character Hub commit: "
            "owner/repo@commit. Local model directories do not require this suffix."
        )
    repo_id, revision = model_id.rsplit("@", 1)
    if not repo_id or not _PINNED_REVISION.fullmatch(revision):
        raise ValueError(
            "Remote Pyrrho model revision must be a 40-character hexadecimal Hub commit."
        )
    return repo_id, revision.lower()


def _preferred_onnx_path(model_dir: Path) -> Path | None:
    """Return the manifest-declared graph, defaulting explicitly to FP32."""
    manifest = _load_release_manifest(model_dir)
    release = manifest.get("release") if manifest is not None else None
    declared = release.get("preferred_onnx_graph") if isinstance(release, dict) else None
    if declared is not None and declared not in ONNX_MODEL_FILENAMES:
        raise ValueError("release.preferred_onnx_graph must be model.onnx or model_quantized.onnx")
    filename = str(declared or DEFAULT_ONNX_MODEL_FILENAME)
    preferred = model_dir / filename
    if preferred.is_file():
        _verify_onnx_parity(model_dir, manifest, preferred)
        return preferred
    if declared is not None:
        raise ValueError(f"manifest-declared Pyrrho ONNX graph is missing: {preferred}")

    # INT8-only packages must opt in through a parity-backed manifest declaration.
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_onnx_parity(
    model_dir: Path,
    manifest: dict[str, Any] | None,
    onnx_path: Path,
) -> None:
    """Bind parity-approved package metadata to the exact selected ONNX bytes."""
    parity = manifest.get("onnx_parity") if manifest is not None else None
    if parity is None:
        if onnx_path.name == "model_quantized.onnx":
            raise ValueError("INT8 Pyrrho selection requires a passed ONNX parity report")
        return
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise ValueError("Pyrrho release ONNX parity is absent or not passed")

    report_name = parity.get("report")
    report_hash = str(parity.get("report_sha256") or "")
    if (
        not isinstance(report_name, str)
        or not report_name
        or Path(report_name).name != report_name
        or not _SHA256.fullmatch(report_hash)
    ):
        raise ValueError("Pyrrho ONNX parity report declaration is invalid")
    report_path = model_dir / report_name
    if not report_path.is_file() or _sha256_file(report_path) != report_hash.lower():
        raise ValueError("Pyrrho ONNX parity report hash mismatch")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Pyrrho ONNX parity report is invalid") from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("Pyrrho ONNX parity report did not pass")

    comparison_name = (
        "native_vs_int8" if onnx_path.name == "model_quantized.onnx" else "native_vs_fp32"
    )
    comparisons = report.get("comparisons")
    comparison = comparisons.get(comparison_name) if isinstance(comparisons, dict) else None
    if not isinstance(comparison, dict) or comparison.get("passed") is not True:
        raise ValueError(f"Pyrrho parity report lacks a passing {comparison_name} comparison")
    artifacts = report.get("artifacts")
    artifact_key = (
        "onnx_int8_sha256" if onnx_path.name == "model_quantized.onnx" else "onnx_fp32_sha256"
    )
    expected_graph_hash = artifacts.get(artifact_key) if isinstance(artifacts, dict) else None
    if not isinstance(expected_graph_hash, str) or not _SHA256.fullmatch(expected_graph_hash):
        raise ValueError("Pyrrho parity report lacks a valid selected-graph hash")
    if _sha256_file(onnx_path) != expected_graph_hash.lower():
        raise ValueError("Pyrrho selected ONNX graph hash differs from its parity report")


def _load_onnx_session(onnx_path: Path) -> Any:
    """Load a Pyrrho ONNX graph with the CPU execution provider."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Pyrrho v2 ONNX inference requires onnxruntime.") from exc
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.get_outputs()
    if not outputs:
        raise ValueError(f"Pyrrho ONNX graph has no outputs: {onnx_path}")
    output_shape = getattr(outputs[0], "shape", None)
    last_dimension = output_shape[-1] if output_shape else None
    if isinstance(last_dimension, int) and last_dimension != _NUM_V2_LABELS:
        raise ValueError(
            f"Pyrrho ONNX graph must expose {_NUM_V2_LABELS} logits; "
            f"got {last_dimension}: {onnx_path}"
        )
    return session


def _load_hf_config(model_dir: Path) -> dict[str, Any]:
    """Read a standard Hugging Face model config."""
    with (model_dir / "config.json").open(encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Pyrrho config must be a JSON object: {model_dir / 'config.json'}")
    return cast(dict[str, Any], loaded)


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
    labels = tuple(
        str(id2label.get(str(index), id2label.get(index, ""))) for index in range(_NUM_V2_LABELS)
    )
    expected_label2id = {label: index for index, label in enumerate(_V2_LABEL_NAMES)}
    return (
        len(id2label) == _NUM_V2_LABELS
        and labels == _V2_LABEL_NAMES
        and config.get("label2id") == expected_label2id
        and config.get("problem_type") == "multi_label_classification"
    )


def _model_cache_dir(model_id: str, *, revision: str | None = None) -> Path:
    """Return Fitz's managed local cache directory for a Pyrrho model."""
    safe_id = model_id.replace("/", "__")
    if revision:
        safe_id = f"{safe_id}__{revision}"
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
    manifest = _load_release_manifest(model_dir)
    if manifest is None:
        return None
    release = manifest.get("release")
    value = release.get("sufficient_threshold") if isinstance(release, dict) else None
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("release.sufficient_threshold must be numeric")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("release.sufficient_threshold must be numeric") from exc
    if not math.isfinite(threshold) or not (1.0 / 3.0) <= threshold <= 1.0:
        raise ValueError("release.sufficient_threshold must be finite and between 1/3 and 1")
    return threshold


def _load_max_length(model_dir: Path) -> int:
    """Read and validate the release token budget against the encoder config."""
    manifest = _load_release_manifest(model_dir)
    release = manifest.get("release") if manifest is not None else None
    raw_value = release.get("max_length") if isinstance(release, dict) else None
    max_length = V2_MAX_LENGTH if raw_value is None else raw_value
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError("release.max_length must be a positive integer")
    if max_length <= 0:
        raise ValueError("release.max_length must be a positive integer")

    config = _load_hf_config(model_dir)
    encoder_limit = config.get("max_position_embeddings")
    if isinstance(encoder_limit, bool) or not isinstance(encoder_limit, int) or encoder_limit <= 0:
        raise ValueError("config.max_position_embeddings must be a positive integer")
    if max_length > encoder_limit:
        raise ValueError(f"release.max_length={max_length} exceeds encoder limit {encoder_limit}")
    return max_length


def _load_release_manifest(model_dir: Path) -> dict[str, Any] | None:
    """Load a package manifest, rejecting malformed release metadata."""
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Pyrrho release manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Pyrrho release manifest must be an object: {manifest_path}")
    if "release" in manifest and not isinstance(manifest["release"], dict):
        raise ValueError(f"Pyrrho release metadata must be an object: {manifest_path}")
    return cast(dict[str, Any], manifest)


def _load_tokenizer(model_dir: Path) -> Any:
    """Load the tokenizer without trusting repo-specific tokenizer metadata."""
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


def _format_evidence_item(item: EvidenceItem) -> str:
    """Return the evidence text Pyrrho should govern over."""
    return str(getattr(item, "content", "") or getattr(item, "excerpt", "") or "")


def _v2_decision_from_logits(
    logits: Any,
    *,
    sufficient_threshold: float | None = TAU,
    input_stats: _InferenceInputStats | None = None,
) -> GovernanceDecision:
    """Convert one v2 output row into the public decision object."""
    evidence_verdict, failure_mode, retrieval_intents, evidence_kinds = _v2_core_heads(
        logits,
        sufficient_threshold=sufficient_threshold,
    )
    evidence_verdict, failure_mode, consistency_reason, pre_consistency_pair = (
        _reconcile_verdict_failure(evidence_verdict, failure_mode)
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
    reason = _reason_for(mode, probs)
    if consistency_reason is not None:
        reason = f"{reason} {consistency_reason}"
    return GovernanceDecision(
        mode=mode,
        probs=probs,
        reason=reason,
        evidence_verdict=evidence_verdict,
        failure_mode=failure_mode,
        retrieval_intents=retrieval_intents,
        evidence_kinds=evidence_kinds,
        heads=heads,
        input_tokens=input_stats.input_tokens if input_stats is not None else None,
        input_truncated=input_stats.input_truncated if input_stats is not None else False,
        max_input_tokens=input_stats.max_input_tokens if input_stats is not None else None,
        used_consistency_fallback=consistency_reason is not None,
        consistency_reason=consistency_reason,
        pre_consistency_pair=pre_consistency_pair,
    )


def _reconcile_verdict_failure(
    evidence_verdict: HeadDecision,
    failure_mode: HeadDecision,
) -> tuple[HeadDecision, HeadDecision, str | None, tuple[str, str]]:
    """Reconcile independent heads to the corpus ontology without upgrading safety."""
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
        # `none` cannot justify a non-sufficient verdict. Preserve or move to the
        # safest verdict and expose an explicit generic insufficiency reason.
        final_verdict = "INSUFFICIENT"
        final_failure = "missing_or_incomplete_evidence"

    if (
        final_verdict == "INSUFFICIENT"
        and final_failure not in _VALID_FAILURES_BY_VERDICT["INSUFFICIENT"]
    ):
        final_failure = "missing_or_incomplete_evidence"

    final_pair = (final_verdict, final_failure)
    reason = (
        "Pyrrho consistency fallback changed contradictory heads "
        f"{original_pair[0]}+{original_pair[1]} to {final_pair[0]}+{final_pair[1]}."
    )
    if final_verdict != evidence_verdict.final_label:
        evidence_verdict = _head_with_consistent_final(
            evidence_verdict,
            final_verdict,
            reason=reason,
        )
    if final_failure != failure_mode.final_label:
        failure_mode = _head_with_consistent_final(
            failure_mode,
            final_failure,
            reason=reason,
        )
    if failure_mode.final_label not in _VALID_FAILURES_BY_VERDICT[evidence_verdict.final_label]:
        raise AssertionError("Pyrrho consistency fallback produced an invalid head pair")
    return evidence_verdict, failure_mode, reason, original_pair


def _head_with_consistent_final(
    head: HeadDecision,
    final_label: str,
    *,
    reason: str,
) -> HeadDecision:
    rebuilt = _head_from_probabilities(
        head.probabilities,
        final_label=final_label,
        raw_label=head.raw_label,
        threshold=head.threshold,
        used_threshold_fallback=head.used_threshold_fallback,
    )
    return replace(
        rebuilt,
        used_consistency_fallback=True,
        consistency_reason=reason,
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
    raw_label = max(probabilities, key=lambda label: probabilities[label])
    if (
        sufficient_threshold is not None
        and "SUFFICIENT" in probabilities
        and raw_label == "SUFFICIENT"
        and probabilities[raw_label] < sufficient_threshold
    ):
        fallback_labels = [label for label in labels if label != "SUFFICIENT"]
        final_label = max(fallback_labels, key=lambda label: probabilities[label])
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
    ranked = sorted(probabilities, key=lambda label: probabilities[label], reverse=True)
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
    ranked = sorted(probabilities, key=lambda label: probabilities[label], reverse=True)
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
