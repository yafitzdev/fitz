# fitz_sage/governance/pyrrho.py
"""
Pyrrho governance backend.

The standard model is ``yafitzdev/pyrrho-v2-nano-g1``: a ModernBERT classifier
with native v2 evidence-verdict, failure-mode, retrieval-intent, and
evidence-kind heads. V2 decisions expose those native heads directly; query-only
v2 planning is inactive until a query-trained v2 head is available.
"""

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
MAX_LENGTH = 4096
MAX_QUERY_LENGTH = 256
V2_MAX_LENGTH = 2048
TAU = 0.34
ONNX_MODEL_FILENAMES = ("model.onnx", "model_quantized.onnx")
_DEFAULT_V2_ALLOW_PATTERNS = (
    "config.json",
    "manifest.json",
    "model.onnx",
    "model_quantized.onnx",
    "ort_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

LABELS = (AnswerMode.ABSTAIN, AnswerMode.DISPUTED, AnswerMode.TRUSTWORTHY)

_V2_VERDICT_LABELS = ("INSUFFICIENT", "DISPUTED", "SUFFICIENT")
_V2_FAILURE_LABELS = (
    "none",
    "unresolved_conflict",
    "missing_or_incomplete_evidence",
    "wrong_scope_or_version",
    "ambiguous_request",
)
_V2_RETRIEVAL_INTENTS = (
    "needs_lookup",
    "needs_temporal_resolution",
    "needs_comparison_or_set",
    "needs_broad_coverage",
)
_V2_EVIDENCE_KINDS = (
    "needs_text",
    "needs_table_or_record",
    "needs_code_or_symbol",
    "needs_config_or_setting",
    "needs_log_or_run_result",
    "needs_document_layout",
)
_V2_VERDICT_TO_MODE = {
    "INSUFFICIENT": AnswerMode.ABSTAIN,
    "DISPUTED": AnswerMode.DISPUTED,
    "SUFFICIENT": AnswerMode.TRUSTWORTHY,
}

_REQUIRED_G5_HEAD_SPECS: dict[str, tuple[str, str]] = {
    "retrieval_action": ("retrieval_action_id2label", "evidence"),
    "gap_type": ("gap_type_id2label", "evidence"),
    "answerability_shape": ("answerability_shape_id2label", "query"),
    "retrieval_modality": ("retrieval_modality_id2label", "query"),
    "retrieval_obligation": ("retrieval_obligation_id2label", "query"),
}


def _head_input_sources(raw: Any) -> dict[str, str]:
    sources = {name: state for name, (_, state) in _REQUIRED_G5_HEAD_SPECS.items()}
    if not isinstance(raw, dict):
        return sources
    for name, source in raw.items():
        if name not in sources:
            continue
        value = str(source)
        if value in {"query", "evidence"}:
            sources[name] = value
    return sources


@dataclass(frozen=True)
class HeadDecision:
    """Prediction metadata for one Pyrrho head."""

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
    """Prediction metadata for one multi-label Pyrrho head."""

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
    """Pyrrho's verdict on a (query, contexts) pair."""

    mode: AnswerMode
    probs: tuple[float, float, float]
    reason: str
    governance: HeadDecision | None = None
    query_contract: HeadDecision | None = None
    route: HeadDecision | None = None
    taxonomy: HeadDecision | None = None
    retrieval_action: HeadDecision | None = None
    gap_type: HeadDecision | None = None
    answerability_shape: HeadDecision | None = None
    retrieval_modality: HeadDecision | None = None
    retrieval_obligation: HeadDecision | None = None
    evidence_verdict: HeadDecision | None = None
    failure_mode: HeadDecision | None = None
    retrieval_intents: MultiLabelDecision | None = None
    evidence_kinds: MultiLabelDecision | None = None
    heads: dict[str, HeadDecision | MultiLabelDecision] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Tuple form for downstream consumers (mirrors the old GovernanceLog API)."""
        return (self.reason,) if self.reason else ()


@dataclass(frozen=True)
class QueryDecision:
    """Pyrrho's query-only retrieval planning signals."""

    query_contract: HeadDecision
    route: HeadDecision
    answerability_shape: HeadDecision
    retrieval_modality: HeadDecision
    retrieval_obligation: HeadDecision
    retrieval_intents: MultiLabelDecision | None = None
    evidence_kinds: MultiLabelDecision | None = None
    heads: dict[str, HeadDecision | MultiLabelDecision] = field(default_factory=dict)


def _format_input(query: str, contexts: Iterable[str]) -> str:
    """Build the full evidence-conditioned text the model was trained on."""
    sources = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, start=1))
    return f"Question: {query}\n\nSources:\n{sources}"


def _format_query_input(query: str) -> str:
    """Build the query-only text for pre-retrieval classification."""
    return f"Question: {query}"


def _reason_for(mode: AnswerMode, probs: tuple[float, float, float]) -> str:
    p_a, p_d, p_t = probs
    if mode is AnswerMode.TRUSTWORTHY:
        return f"Pyrrho: sources support a confident answer (P={p_t:.2f})."
    if mode is AnswerMode.DISPUTED:
        return f"Pyrrho: sources disagree on the answer (P={p_d:.2f})."
    return f"Pyrrho: retrieved sources do not contain enough evidence (P={p_a:.2f})."


def _load_g5_torch_model(model_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the legacy g5 multitask architecture through optional Torch deps."""
    try:
        import torch
        from safetensors.torch import load_file
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Legacy Pyrrho g5 packages require optional Torch dependencies. "
            "Install fitz-sage with the 'legacy-pyrrho' extra or use a v2 ONNX package."
        ) from exc

    from transformers import AutoConfig, AutoModel

    class _PyrrhoMultiTaskModernBert(nn.Module):
        """Package-local copy of the legacy Pyrrho multitask architecture."""

        def __init__(self, package_dir: Path) -> None:
            super().__init__()
            self.pyrrho_config = _load_multitask_config(package_dir)
            self._head_input_sources = _head_input_sources(
                self.pyrrho_config.get("head_input_sources")
            )
            backbone_config = AutoConfig.from_pretrained(package_dir)
            self.backbone = AutoModel.from_config(backbone_config)
            hidden_size = int(backbone_config.hidden_size)
            self.dropout = nn.Dropout(float(self.pyrrho_config.get("dropout", 0.0)))
            self.governance_head = nn.Linear(
                hidden_size,
                int(self.pyrrho_config["num_governance_labels"]),
            )
            self.query_contract_head = nn.Linear(
                hidden_size,
                int(self.pyrrho_config["num_query_contract_labels"]),
            )
            self.route_head = nn.Linear(hidden_size, int(self.pyrrho_config["num_routes"]))
            self.taxonomy_head = nn.Linear(
                hidden_size,
                int(self.pyrrho_config["num_taxonomy_patterns"]),
            )
            self.scalar_head = nn.Linear(
                hidden_size,
                len(self.pyrrho_config["scalar_fields"]),
            )
            for name, (label_key, _) in _REQUIRED_G5_HEAD_SPECS.items():
                labels = _required_label_map(self.pyrrho_config, label_key)
                setattr(self, f"{name}_head", nn.Linear(hidden_size, len(labels)))
            self.load_state_dict(load_file(package_dir / "model.safetensors", device="cpu"))

        @staticmethod
        def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
            mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        def _encode(self, input_ids: Any, attention_mask: Any) -> Any:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return self.dropout(self._mean_pool(outputs.last_hidden_state, attention_mask))

        def forward(
            self,
            *,
            input_ids: Any,
            attention_mask: Any,
            query_input_ids: Any,
            query_attention_mask: Any,
        ) -> dict[str, Any]:
            evidence_state = self._encode(input_ids, attention_mask)
            query_state = self._encode(query_input_ids, query_attention_mask)
            outputs = {
                "governance_logits": self.governance_head(evidence_state),
                "query_contract_logits": self.query_contract_head(query_state),
                "route_logits": self.route_head(query_state),
                "taxonomy_logits": self.taxonomy_head(evidence_state),
                "scalar_preds": torch.sigmoid(self.scalar_head(evidence_state)),
            }
            for name, (_, state_source) in _REQUIRED_G5_HEAD_SPECS.items():
                head = getattr(self, f"{name}_head")
                state = (
                    query_state
                    if self._head_input_sources.get(name, state_source) == "query"
                    else evidence_state
                )
                outputs[f"{name}_logits"] = head(state)
            return outputs

    model = _PyrrhoMultiTaskModernBert(model_dir).eval()
    return model, model.pyrrho_config


class Pyrrho:
    """The Pyrrho governance classifier."""

    supports_batched_prefixes = True

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._model_kind = "g5"
        self._model_dir: Path | None = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._runtime = ""
        self._id2label: dict[int, str] = {}
        self._query_contract_id2label: dict[int, str] = {}
        self._route_id2label: dict[int, str] = {}
        self._taxonomy_id2label: dict[int, str] = {}
        self._g5_head_id2labels: dict[str, dict[int, str]] = {}
        self._scalar_fields: tuple[str, ...] = ()
        self._trustworthy_threshold = TAU

    def classify_query(self, query: str) -> QueryDecision:
        """Classify query-only retrieval planning signals before retrieval."""
        return self._predict_query(query)

    def decide(self, query: str, contexts: list[EvidenceItem]) -> GovernanceDecision:
        """Classify a (query, contexts) pair into one of the governance modes."""
        return self.decide_many(query, [contexts])[0]

    def decide_many(
        self,
        query: str,
        contexts_by_prefix: list[list[EvidenceItem]],
    ) -> list[GovernanceDecision]:
        """Classify several evidence prefixes in one model batch."""
        decisions: list[GovernanceDecision | None] = [None] * len(contexts_by_prefix)
        raw_non_empty_contexts: list[list[EvidenceItem]] = []
        non_empty_positions: list[int] = []

        for index, contexts in enumerate(contexts_by_prefix):
            if not contexts:
                decisions[index] = GovernanceDecision(
                    mode=AnswerMode.ABSTAIN,
                    probs=(1.0, 0.0, 0.0),
                    reason="Pyrrho: no contexts retrieved.",
                )
                continue
            non_empty_positions.append(index)
            raw_non_empty_contexts.append(contexts)

        if raw_non_empty_contexts:
            self._load()
            formatter = (
                _format_v2_evidence_item
                if getattr(self, "_model_kind", "g5") == "v2_alpha"
                else _format_evidence_item
            )
            non_empty_contexts = [
                [formatter(context) for context in contexts]
                for contexts in raw_non_empty_contexts
            ]
            for index, decision in zip(
                non_empty_positions,
                self._predict_context_batches(query, non_empty_contexts),
                strict=True,
            ):
                decisions[index] = decision

        return [decision for decision in decisions if decision is not None]

    def _load(self) -> None:
        """Load the tokenizer and custom multitask checkpoint once per process."""
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
            self._model_dir = model_dir
            if (model_dir / "pyrrho_multitask_config.json").exists():
                model, config = _load_g5_torch_model(model_dir)

                self._model_kind = "g5"
                self._runtime = "torch"
                self._tokenizer = _load_tokenizer(model_dir)
                self._model = model
                self._id2label = _id2label(config["id2label"])
                self._query_contract_id2label = _id2label(config["query_contract_id2label"])
                self._route_id2label = _id2label(config["route_id2label"])
                self._taxonomy_id2label = _id2label(config["taxonomy_id2label"])
                self._g5_head_id2labels = {
                    name: _id2label(_required_label_map(config, label_key))
                    for name, (label_key, _) in _REQUIRED_G5_HEAD_SPECS.items()
                }
                self._scalar_fields = tuple(str(field) for field in config["scalar_fields"])
                trustworthy_threshold = _load_trustworthy_threshold(model_dir)
                self._trustworthy_threshold = (
                    TAU if trustworthy_threshold is None else trustworthy_threshold
                )
                return

            if _is_v2_alpha_model_dir(model_dir):
                self._model_kind = "v2_alpha"
                self._tokenizer = _load_tokenizer(model_dir)
                onnx_path = _preferred_onnx_path(model_dir)
                if onnx_path is not None:
                    self._runtime = "onnx"
                    self._model = _load_onnx_session(onnx_path)
                else:
                    self._runtime = "torch"
                    self._model = _load_v2_torch_model(model_dir)
                self._id2label = _id2label(_load_hf_config(model_dir)["id2label"])
                return

            raise ValueError(
                f"Pyrrho model directory is neither a g5 multitask package nor a v2 "
                f"alpha classifier: {model_dir}"
            )

    def _predict_query(self, query: str) -> QueryDecision:
        """Run query-only heads for one query."""
        self._load()
        if getattr(self, "_model_kind", "g5") == "v2_alpha":
            return _v2_query_decision_from_logits(None)

        outputs = self._run_batch(
            full_texts=[_format_input(query, ())],
            query_texts=[_format_query_input(query)],
        )
        query_contract = _head_decision(
            outputs["query_contract_logits"][0],
            self._query_contract_id2label,
        )
        route = _head_decision(outputs["route_logits"][0], self._route_id2label)
        g5_query_heads = {
            name: _head_decision(outputs[f"{name}_logits"][0], id2label)
            for name, id2label in self._g5_head_id2labels.items()
            if _REQUIRED_G5_HEAD_SPECS[name][1] == "query"
        }
        heads = {
            "query_contract": query_contract,
            "route": route,
            **g5_query_heads,
        }
        return QueryDecision(
            query_contract=query_contract,
            route=route,
            answerability_shape=g5_query_heads["answerability_shape"],
            retrieval_modality=g5_query_heads["retrieval_modality"],
            retrieval_obligation=g5_query_heads["retrieval_obligation"],
            heads=heads,
        )

    def _predict_context_batches(
        self,
        query: str,
        contexts_by_prefix: list[list[str]],
    ) -> list[GovernanceDecision]:
        """Run all non-empty evidence prefixes in one model batch."""
        self._load()
        if getattr(self, "_model_kind", "g5") == "v2_alpha":
            logits = self._run_v2_texts(
                [_format_input(query, contexts) for contexts in contexts_by_prefix]
            )
            return [_v2_decision_from_logits(row) for row in logits]

        outputs = self._run_batch(
            full_texts=[_format_input(query, contexts) for contexts in contexts_by_prefix],
            query_texts=[_format_query_input(query)] * len(contexts_by_prefix),
        )
        decisions: list[GovernanceDecision] = []
        for index in range(len(contexts_by_prefix)):
            decisions.append(_decision_from_outputs(self, outputs, index))
        return decisions

    def _run_batch(self, *, full_texts: list[str], query_texts: list[str]) -> dict[str, Any]:
        """Tokenize and run a Pyrrho multitask batch."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Pyrrho was not loaded before inference.")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Legacy Pyrrho g5 inference requires optional Torch dependencies. "
                "Install fitz-sage with the 'legacy-pyrrho' extra."
            ) from exc

        full = self._tokenizer(
            full_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        query = self._tokenizer(
            query_texts,
            truncation=True,
            max_length=MAX_QUERY_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return self._model(
                input_ids=full["input_ids"],
                attention_mask=full["attention_mask"],
                query_input_ids=query["input_ids"],
                query_attention_mask=query["attention_mask"],
            )

    def _run_v2_texts(self, texts: list[str]) -> Any:
        """Tokenize and run a v2 alpha classifier batch."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Pyrrho was not loaded before inference.")

        if self._runtime == "onnx":
            return self._run_v2_onnx_texts(texts)
        return self._run_v2_torch_texts(texts)

    def _run_v2_onnx_texts(self, texts: list[str]) -> Any:
        """Tokenize and run a v2 classifier batch through ONNX Runtime."""
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

    def _run_v2_torch_texts(self, texts: list[str]) -> Any:
        """Tokenize and run a v2 classifier batch through a Torch fallback."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "This Pyrrho v2 package has no ONNX graph and requires optional Torch "
                "dependencies for fallback inference."
            ) from exc

        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=V2_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return self._model(**encoded).logits


def _load_multitask_config(model_dir: Path) -> dict[str, Any]:
    """Read the pyrrho multitask package config."""
    with (model_dir / "pyrrho_multitask_config.json").open(encoding="utf-8") as f:
        return json.load(f)


def _snapshot_allow_patterns(model_id: str) -> tuple[str, ...] | None:
    """Limit the default v2 download to files needed by the ONNX runtime."""
    if model_id == MODEL_ID:
        return _DEFAULT_V2_ALLOW_PATTERNS
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
        raise RuntimeError(
            "Pyrrho v2 ONNX inference requires onnxruntime. Install fitz-sage with "
            "its standard runtime dependencies."
        ) from exc
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _load_v2_torch_model(model_dir: Path) -> Any:
    """Load a v2 classifier through optional Torch fallback dependencies."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "This Pyrrho v2 package does not include model.onnx or "
            "model_quantized.onnx, and Torch is not installed for fallback "
            "safetensors inference."
        ) from exc

    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(model_dir).eval()


def _load_hf_config(model_dir: Path) -> dict[str, Any]:
    """Read a standard Hugging Face model config."""
    with (model_dir / "config.json").open(encoding="utf-8") as f:
        return json.load(f)


def _is_v2_alpha_model_dir(model_dir: Path) -> bool:
    """Return whether a directory is a v2 alpha 18-logit classifier package."""
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


def _required_label_map(config: dict[str, Any], key: str) -> dict[str, str]:
    """Read a required g5 label map and fail fast when a package is stale."""
    labels = config.get(key)
    if not isinstance(labels, dict) or not labels:
        raise ValueError(
            f"Pyrrho g5 package is required; missing non-empty {key!r} in "
            "pyrrho_multitask_config.json."
        )
    return labels


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


def _load_trustworthy_threshold(model_dir: Path) -> float | None:
    """Read the packaged release threshold when a Pyrrho manifest provides one."""
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = manifest.get("release", {}).get("trustworthy_threshold")
        return float(value) if value is not None else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_tokenizer(model_dir: Path) -> Any:
    """Load the tokenizer without trusting the repo's tokenizer_class metadata."""
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
    """Return evidence content plus a compact ledger Pyrrho can govern over."""
    content = str(getattr(item, "content", "") or "")
    metadata = getattr(item, "metadata", None)
    if not isinstance(metadata, dict):
        return content

    ledger_lines = _evidence_ledger_lines(metadata)
    if not ledger_lines:
        return content
    return "\n".join(
        [
            content,
            "",
            "Pyrrho evidence ledger:",
            *ledger_lines,
        ]
    )


def _format_v2_evidence_item(item: EvidenceItem) -> str:
    """Return v2 evidence text without fitz-sage's runtime ledger."""
    return str(getattr(item, "content", "") or getattr(item, "excerpt", "") or "")


def _evidence_ledger_lines(metadata: dict[str, Any]) -> list[str]:
    """Serialize retrieval/compiler facts as bounded natural-language lines."""
    lines: list[str] = []

    compiler = metadata.get("evidence_compiler")
    if isinstance(compiler, dict):
        roles = _bounded_values(compiler.get("roles"))
        if roles:
            lines.append(f"- compiler roles: {', '.join(roles)}")
        min_sources = compiler.get("min_sources")
        if min_sources is not None:
            lines.append(f"- compiler minimum sources: {min_sources}")
        content_scope = compiler.get("content_scope")
        if content_scope:
            lines.append(f"- compiler content scope: {content_scope}")
        contract = compiler.get("contract")
        if isinstance(contract, dict):
            for key in (
                "identifiers",
                "phrase_anchors",
                "source_anchors",
                "required_modalities",
                "temporal_policy",
            ):
                values = _bounded_values(contract.get(key))
                if values:
                    lines.append(f"- contract {key}: {', '.join(values)}")

    closure = metadata.get("evidence_closure")
    if isinstance(closure, dict):
        role = closure.get("role")
        if role:
            lines.append(f"- closure role: {role}")
        reason = closure.get("reason")
        if reason:
            lines.append(f"- closure reason: {reason}")
        bridges = _bounded_values(closure.get("bridges"))
        if bridges:
            lines.append(f"- closure bridges: {', '.join(bridges)}")

    table_plan = metadata.get("table_query_plan")
    if isinstance(table_plan, dict):
        identifiers = _bounded_values(table_plan.get("identifiers"))
        if identifiers:
            lines.append(f"- table identifiers: {', '.join(identifiers)}")
        predicates = _bounded_values(table_plan.get("predicates"))
        if predicates:
            lines.append(f"- table predicates: {', '.join(predicates)}")
        sort = _bounded_values(table_plan.get("sort"))
        if sort:
            lines.append(f"- table sort: {', '.join(sort)}")

    return lines[:16]


def _bounded_values(value: Any, *, limit: int = 8) -> list[str]:
    """Return short serializable ledger values."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, dict):
        values = [f"{key}={item}" for key, item in value.items()]
    elif isinstance(value, Iterable):
        values = [str(item) for item in value]
    else:
        values = [str(value)]
    bounded: list[str] = []
    for item in values:
        normalized = " ".join(str(item).split())
        if not normalized:
            continue
        bounded.append(normalized[:120])
        if len(bounded) >= limit:
            break
    return bounded


def _id2label(raw: dict[str, str]) -> dict[int, str]:
    """Normalize JSON id->label mappings."""
    return {int(key): str(value) for key, value in raw.items()}


def _v2_decision_from_logits(logits: Any) -> GovernanceDecision:
    """Convert one v2 output row into the public decision object."""
    evidence_verdict, failure_mode, retrieval_intents, evidence_kinds = _v2_core_heads(logits)
    probs = (
        float(evidence_verdict.probabilities["INSUFFICIENT"]),
        float(evidence_verdict.probabilities["DISPUTED"]),
        float(evidence_verdict.probabilities["SUFFICIENT"]),
    )
    mode = _V2_VERDICT_TO_MODE[evidence_verdict.final_label]
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


def _v2_query_decision_from_logits(logits: Any) -> QueryDecision:
    """Return inactive query planning for v2.

    The v2 g1 checkpoint was trained and validated on evidence-conditioned
    ``Question + Sources`` rows. Query-only execution can produce overconfident
    retrieval labels that hurt fitz-sage's first retrieval pass, so runtime
    planning stays inactive until a query-trained v2 head exists.
    """
    _ = logits
    inactive = _inactive_head()
    return QueryDecision(
        query_contract=inactive,
        route=inactive,
        answerability_shape=inactive,
        retrieval_modality=inactive,
        retrieval_obligation=inactive,
        heads={},
    )


def _v2_core_heads(
    logits: Any,
) -> tuple[HeadDecision, HeadDecision, MultiLabelDecision, MultiLabelDecision]:
    """Decode the four native v2 output groups."""
    evidence_verdict = _single_label_decision(logits, 0, _V2_VERDICT_LABELS)
    failure_mode = _single_label_decision(logits, 3, _V2_FAILURE_LABELS)
    retrieval_intents = _multi_label_decision(logits, 8, _V2_RETRIEVAL_INTENTS)
    evidence_kinds = _multi_label_decision(logits, 12, _V2_EVIDENCE_KINDS)
    return evidence_verdict, failure_mode, retrieval_intents, evidence_kinds


def _single_label_decision(
    logits: Any,
    start: int,
    labels: tuple[str, ...],
) -> HeadDecision:
    """Decode a mutually exclusive v2 label group."""
    probabilities = {
        label: probability
        for label, probability in zip(labels, _softmax(_slice_logits(logits, start, len(labels))))
    }
    return _head_from_probabilities(probabilities)


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
        entropy=float(
            -sum(prob * math.log(prob) for prob in probabilities.values() if prob > 0.0)
        ),
    )


def _inactive_head() -> HeadDecision:
    """Build a no-op head for unavailable query-only signals."""
    return HeadDecision(
        raw_label="",
        final_label="",
        used_threshold_fallback=False,
        threshold=None,
        confidence=0.0,
        probabilities={},
        runner_up_label="",
        runner_up_probability=0.0,
        margin_to_runner_up=0.0,
        entropy=0.0,
    )


def _slice_logits(logits: Any, start: int, size: int) -> list[float]:
    """Return a flat slice from a tensor/list output row."""
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


def _decision_from_outputs(
    pyrrho: Pyrrho,
    outputs: dict[str, Any],
    index: int,
) -> GovernanceDecision:
    """Convert one model output row into the public decision object."""
    governance = _head_decision(
        outputs["governance_logits"][index],
        pyrrho._id2label,
        trustworthy_threshold=pyrrho._trustworthy_threshold,
    )
    query_contract = _head_decision(
        outputs["query_contract_logits"][index],
        pyrrho._query_contract_id2label,
    )
    route = _head_decision(outputs["route_logits"][index], pyrrho._route_id2label)
    taxonomy = _head_decision(
        outputs["taxonomy_logits"][index],
        pyrrho._taxonomy_id2label,
    )
    g5_heads = {
        name: _head_decision(outputs[f"{name}_logits"][index], id2label)
        for name, id2label in pyrrho._g5_head_id2labels.items()
    }
    heads = {
        "governance": governance,
        "query_contract": query_contract,
        "route": route,
        "taxonomy": taxonomy,
        **g5_heads,
    }
    scalars = {
        field: float(value)
        for field, value in zip(
            pyrrho._scalar_fields,
            _to_float_list(outputs["scalar_preds"][index]),
            strict=True,
        )
    }

    probs = (
        float(governance.probabilities["ABSTAIN"]),
        float(governance.probabilities["DISPUTED"]),
        float(governance.probabilities["TRUSTWORTHY"]),
    )
    mode = AnswerMode(governance.final_label.lower())
    return GovernanceDecision(
        mode=mode,
        probs=probs,
        reason=_reason_for(mode, probs),
        governance=governance,
        query_contract=query_contract,
        route=route,
        taxonomy=taxonomy,
        retrieval_action=g5_heads["retrieval_action"],
        gap_type=g5_heads["gap_type"],
        answerability_shape=g5_heads["answerability_shape"],
        retrieval_modality=g5_heads["retrieval_modality"],
        retrieval_obligation=g5_heads["retrieval_obligation"],
        heads=heads,
        scalars=scalars,
    )


def _head_decision(
    logits: Any,
    id2label: dict[int, str],
    *,
    trustworthy_threshold: float | None = None,
) -> HeadDecision:
    """Build rich metadata for one classifier head."""
    probs = _softmax(_to_float_list(logits))
    ranked_ids = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
    raw_id = ranked_ids[0]
    final_id = raw_id
    used_threshold_fallback = False

    if (
        trustworthy_threshold is not None
        and id2label.get(raw_id) == "TRUSTWORTHY"
        and probs[raw_id] < trustworthy_threshold
    ):
        fallback_ids = [
            idx for idx in range(len(probs)) if id2label.get(idx) != "TRUSTWORTHY"
        ]
        final_id = max(fallback_ids, key=lambda idx: probs[idx])
        used_threshold_fallback = True

    runner_up_id = next((idx for idx in ranked_ids if idx != final_id), final_id)
    probabilities = {id2label[idx]: float(probs[idx]) for idx in range(len(probs))}
    confidence = float(probs[final_id])
    runner_up_probability = float(probs[runner_up_id])
    return HeadDecision(
        raw_label=id2label[raw_id],
        final_label=id2label[final_id],
        used_threshold_fallback=used_threshold_fallback,
        threshold=trustworthy_threshold,
        confidence=confidence,
        probabilities=probabilities,
        runner_up_label=id2label[runner_up_id],
        runner_up_probability=runner_up_probability,
        margin_to_runner_up=float(confidence - runner_up_probability),
        entropy=float(-sum(prob * math.log(prob) for prob in probs if prob > 0.0)),
    )


def _softmax(logits: list[float]) -> list[float]:
    """Softmax one logits row with numerical stabilization."""
    offset = max(logits)
    exp = [math.exp(float(value) - offset) for value in logits]
    total = sum(exp)
    return [value / total for value in exp]


__all__ = [
    "GovernanceDecision",
    "HeadDecision",
    "MODEL_ID",
    "MultiLabelDecision",
    "Pyrrho",
    "QueryDecision",
    "TAU",
]
