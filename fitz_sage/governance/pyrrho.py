# fitz_sage/governance/pyrrho.py
"""
Pyrrho governance backend.

The standard model is ``yafitzdev/pyrrho-nano-g3.1``: a custom multitask
ModernBERT encoder with governance, query-contract, route/domain, taxonomy,
and scalar heads. It is a local CPU model loaded from Hugging Face; no LLM call
is made on the governance path.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file
from torch import nn

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.core.paths import FitzPaths
from fitz_sage.governance.protocol import EvidenceItem

MODEL_ID = "yafitzdev/pyrrho-nano-g3.1"
MAX_LENGTH = 4096
MAX_QUERY_LENGTH = 256
TAU = 0.39

LABELS = (AnswerMode.ABSTAIN, AnswerMode.DISPUTED, AnswerMode.TRUSTWORTHY)


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
class GovernanceDecision:
    """Pyrrho's verdict on a (query, contexts) pair."""

    mode: AnswerMode
    probs: tuple[float, float, float]
    reason: str
    governance: HeadDecision | None = None
    query_contract: HeadDecision | None = None
    route: HeadDecision | None = None
    taxonomy: HeadDecision | None = None
    scalars: dict[str, float] = field(default_factory=dict)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Tuple form for downstream consumers (mirrors the old GovernanceLog API)."""
        return (self.reason,) if self.reason else ()


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


class _PyrrhoMultiTaskModernBert(nn.Module):
    """Package-local copy of the g3.1 multitask architecture."""

    def __init__(self, model_dir: Path) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.pyrrho_config = _load_multitask_config(model_dir)
        backbone_config = AutoConfig.from_pretrained(model_dir)
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
        self.scalar_head = nn.Linear(hidden_size, len(self.pyrrho_config["scalar_fields"]))
        self.load_state_dict(load_file(model_dir / "model.safetensors", device="cpu"))

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
        return {
            "governance_logits": self.governance_head(evidence_state),
            "query_contract_logits": self.query_contract_head(query_state),
            "route_logits": self.route_head(query_state),
            "taxonomy_logits": self.taxonomy_head(evidence_state),
            "scalar_preds": torch.sigmoid(self.scalar_head(evidence_state)),
        }


class Pyrrho:
    """The pyrrho g3.1 classifier."""

    supports_batched_prefixes = True

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._lock = threading.Lock()
        self._model_dir: Path | None = None
        self._tokenizer: Any = None
        self._model: _PyrrhoMultiTaskModernBert | None = None
        self._id2label: dict[int, str] = {}
        self._query_contract_id2label: dict[int, str] = {}
        self._route_id2label: dict[int, str] = {}
        self._taxonomy_id2label: dict[int, str] = {}
        self._scalar_fields: tuple[str, ...] = ()

    def classify_query(self, query: str) -> HeadDecision:
        """Classify the query contract before retrieval."""
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
        non_empty_contexts: list[list[str]] = []
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
            non_empty_contexts.append([context.content for context in contexts])

        if non_empty_contexts:
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

            from huggingface_hub import snapshot_download

            model_dir = _model_cache_dir(self._model_id)
            model_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=self._model_id,
                local_dir=model_dir,
            )
            model = _PyrrhoMultiTaskModernBert(model_dir).eval()
            config = model.pyrrho_config

            self._model_dir = model_dir
            self._tokenizer = _load_tokenizer(model_dir)
            self._model = model
            self._id2label = _id2label(config["id2label"])
            self._query_contract_id2label = _id2label(config["query_contract_id2label"])
            self._route_id2label = _id2label(config["route_id2label"])
            self._taxonomy_id2label = _id2label(config["taxonomy_id2label"])
            self._scalar_fields = tuple(str(field) for field in config["scalar_fields"])

    @torch.no_grad()
    def _predict_query(self, query: str) -> HeadDecision:
        """Run query-only heads for one query."""
        self._load()
        outputs = self._run_batch(
            full_texts=[_format_input(query, ())],
            query_texts=[_format_query_input(query)],
        )
        return _head_decision(
            outputs["query_contract_logits"][0],
            self._query_contract_id2label,
        )

    @torch.no_grad()
    def _predict_context_batches(
        self,
        query: str,
        contexts_by_prefix: list[list[str]],
    ) -> list[GovernanceDecision]:
        """Run all non-empty evidence prefixes in one model batch."""
        self._load()
        outputs = self._run_batch(
            full_texts=[_format_input(query, contexts) for contexts in contexts_by_prefix],
            query_texts=[_format_query_input(query)] * len(contexts_by_prefix),
        )
        decisions: list[GovernanceDecision] = []
        for index in range(len(contexts_by_prefix)):
            decisions.append(_decision_from_outputs(self, outputs, index))
        return decisions

    def _run_batch(self, *, full_texts: list[str], query_texts: list[str]) -> dict[str, Any]:
        """Tokenize and run a g3.1 multitask batch."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Pyrrho was not loaded before inference.")

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
        return self._model(
            input_ids=full["input_ids"],
            attention_mask=full["attention_mask"],
            query_input_ids=query["input_ids"],
            query_attention_mask=query["attention_mask"],
        )


def _load_multitask_config(model_dir: Path) -> dict[str, Any]:
    """Read the pyrrho multitask package config."""
    with (model_dir / "pyrrho_multitask_config.json").open(encoding="utf-8") as f:
        return json.load(f)


def _model_cache_dir(model_id: str) -> Path:
    """Return Fitz's managed local cache directory for a Pyrrho model."""
    safe_id = model_id.replace("/", "__")
    return FitzPaths.user_home() / "models" / "pyrrho" / safe_id


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


def _id2label(raw: dict[str, str]) -> dict[int, str]:
    """Normalize JSON id->label mappings."""
    return {int(key): str(value) for key, value in raw.items()}


def _decision_from_outputs(
    pyrrho: Pyrrho,
    outputs: dict[str, Any],
    index: int,
) -> GovernanceDecision:
    """Convert one model output row into the public decision object."""
    governance = _head_decision(
        outputs["governance_logits"][index],
        pyrrho._id2label,
        trustworthy_threshold=TAU,
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
    scalars = {
        field: float(value)
        for field, value in zip(
            pyrrho._scalar_fields,
            outputs["scalar_preds"][index].detach().cpu().tolist(),
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
        scalars=scalars,
    )


def _head_decision(
    logits: Any,
    id2label: dict[int, str],
    *,
    trustworthy_threshold: float | None = None,
) -> HeadDecision:
    """Build rich metadata for one classifier head."""
    probs = _softmax(logits.detach().cpu().tolist())
    ranked_ids = sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)
    raw_id = ranked_ids[0]
    final_id = raw_id
    used_threshold_fallback = False

    if (
        trustworthy_threshold is not None
        and id2label.get(raw_id) == "TRUSTWORTHY"
        and probs[raw_id] < trustworthy_threshold
    ):
        fallback_ids = [idx for idx in range(len(probs)) if id2label.get(idx) != "TRUSTWORTHY"]
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
    "Pyrrho",
    "TAU",
]
