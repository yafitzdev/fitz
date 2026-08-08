"""Benchmark-only controls for isolating query expansion and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressRerankResponse
from fitz_sage.engines.fitz_krag.retrieval.trace import addresses_trace

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.types import Address


@dataclass(frozen=True)
class RetrievalAblation:
    """One explicit query-side component configuration."""

    name: str
    semantic_expansion: bool
    cross_encoder_reranking: bool
    description: str

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "semantic_expansion": self.semantic_expansion,
            "cross_encoder_reranking": self.cross_encoder_reranking,
            "description": self.description,
        }


ABLATIONS = {
    "literal": RetrievalAblation(
        name="literal",
        semantic_expansion=False,
        cross_encoder_reranking=False,
        description=(
            "Deterministic Fitz-Sage query planning and typed lexical recall; "
            "managed Qwen keywords are disabled and cross-encoder scoring is "
            "replaced by stable top-k selection."
        ),
    ),
    "expansion": RetrievalAblation(
        name="expansion",
        semantic_expansion=True,
        cross_encoder_reranking=False,
        description=(
            "Literal variant plus managed Qwen semantic query keywords; "
            "cross-encoder scoring is replaced by stable top-k selection."
        ),
    ),
    "reranker": RetrievalAblation(
        name="reranker",
        semantic_expansion=False,
        cross_encoder_reranking=True,
        description=(
            "Literal variant plus the canonical INT8 cross-encoder reranker; "
            "managed Qwen semantic query keywords are disabled."
        ),
    ),
    "full": RetrievalAblation(
        name="full",
        semantic_expansion=True,
        cross_encoder_reranking=True,
        description="Canonical Fitz-Sage query pipeline with Qwen keywords and reranking.",
    ),
}


def ablation_names() -> tuple[str, ...]:
    return tuple(ABLATIONS)


def get_ablation(name: str) -> RetrievalAblation:
    """Resolve a named benchmark configuration."""
    try:
        return ABLATIONS[name]
    except KeyError as exc:
        choices = ", ".join(ABLATIONS)
        raise ValueError(f"Unknown retrieval ablation {name!r}; choose one of: {choices}") from exc


def apply_ablation(engine: Any, ablation: RetrievalAblation) -> None:
    """Apply query-only controls after canonical engine construction."""
    query_pipeline = _required_attribute(engine, "_query_pipeline")
    retrieval_pass = _required_attribute(engine, "_retrieval_pass")

    if not ablation.semantic_expansion:
        _required_attribute(query_pipeline, "_semantic_keyword_batcher")
        engine._semantic_keyword_batcher = None
        query_pipeline._semantic_keyword_batcher = None

    if not ablation.cross_encoder_reranking:
        _required_attribute(retrieval_pass, "_reranker")
        config = _required_attribute(engine, "_config")
        selector = StableTopKSelector(k=int(config.rerank_k))
        engine._address_reranker = selector
        retrieval_pass._reranker = selector


class StableTopKSelector:
    """Preserve the reranker's output budget without cross-encoder scoring."""

    def __init__(self, *, k: int) -> None:
        if k < 1:
            raise ValueError("Stable top-k selection requires a positive limit.")
        self._k = k

    def rerank(
        self,
        query: str,
        addresses: list[Address],
    ) -> AddressRerankResponse:
        selected = addresses[: self._k]
        return AddressRerankResponse(
            addresses=selected,
            trace={
                "used": False,
                "reason": "benchmark_top_k_passthrough",
                "query": query,
                "input_count": len(addresses),
                "output_count": len(selected),
                "output": addresses_trace(selected),
            },
        )


def _required_attribute(value: Any, name: str) -> Any:
    if not hasattr(value, name):
        raise RuntimeError(
            f"BEIR ablation expected {type(value).__name__}.{name}; "
            "update the benchmark control for the current engine structure."
        )
    return getattr(value, name)
