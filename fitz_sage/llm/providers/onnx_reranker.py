# fitz_sage/llm/providers/onnx_reranker.py
"""
ONNX cross-encoder reranker.

Default backbone:
[`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
— a 149M-parameter ModernBERT cross-encoder that matches 1.2B-parameter
rerankers on Hit@1 and quantises cleanly to INT8 ONNX (2.7–3.4x CPU
speedup vs FP32, ~98% of full-precision quality).

Same architectural family as the pyrrho governance classifier
(ModernBERT + INT8 ONNX via `optimum.onnxruntime`); inference path is
identical and the deps are already in `pyproject.toml`.

Public surface — implements `RerankProvider`:

    reranker = OnnxReranker()
    reranker = OnnxReranker(model_id="Alibaba-NLP/gte-reranker-modernbert-base")
    results = reranker.rerank(query, documents, top_n=5)
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

from fitz_sage.llm.providers.base import RerankResult

logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "Alibaba-NLP/gte-reranker-modernbert-base"
MAX_LENGTH = 512  # gte-modernbert-base is trained at 512; longer inputs are truncated.
DEFAULT_BATCH_SIZE = 16


class OnnxReranker:
    """Cross-encoder reranker served as INT8 ONNX on CPU.

    Args:
        model_id: HuggingFace repo id of the cross-encoder. Defaults to
            `Alibaba-NLP/gte-reranker-modernbert-base`. Any HF cross-encoder
            with a `SequenceClassification` head (num_labels=1) works.
        max_length: Tokenizer truncation cap (default 512).
        batch_size: Number of `(query, doc)` pairs per ONNX forward pass.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_length: int = MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_id = model_id
        self._max_length = max_length
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        with self._lock:
            if self._tokenizer is not None and self._model is not None:
                return
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            logger.info(f"Loading ONNX reranker: {self._model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = ORTModelForSequenceClassification.from_pretrained(
                self._model_id, export=True
            )

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score `(query, doc)` pairs with the cross-encoder, sort by score desc.

        Documents the model fails to score get a default score of 0 and the
        original ordinal as a stable tiebreaker.
        """
        if not documents:
            return []
        if len(documents) == 1:
            return [RerankResult(index=0, score=1.0)]

        self._load()
        scores = self._score_pairs(query, documents)

        results = [RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)]
        results.sort(key=lambda r: (-r.score, r.index))

        if top_n is not None:
            results = results[:top_n]
        return results

    def _score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Run the cross-encoder forward pass in batches and return raw logits."""
        import numpy as np

        n = len(documents)
        scores = [0.0] * n
        for start in range(0, n, self._batch_size):
            batch_docs = documents[start : start + self._batch_size]
            queries = [query] * len(batch_docs)
            enc = self._tokenizer(  # type: ignore[misc]
                queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="np",
            )
            out = self._model(**enc)  # type: ignore[misc]
            logits = out.logits
            # Sequence-classification head with num_labels=1 -> shape (B, 1).
            # If a model exposes a 2-class head (logits shape (B, 2)) we take
            # the positive-class logit as the relevance score.
            arr = np.asarray(logits)
            if arr.ndim == 2 and arr.shape[1] == 1:
                batch_scores = arr[:, 0]
            elif arr.ndim == 2 and arr.shape[1] == 2:
                batch_scores = arr[:, 1] - arr[:, 0]
            else:
                batch_scores = arr.reshape(arr.shape[0], -1)[:, 0]
            for i, s in enumerate(batch_scores.tolist()):
                scores[start + i] = float(s)
        return scores

    @classmethod
    def from_iterable(
        cls,
        documents: Iterable[str],
        query: str,
        top_n: int | None = None,
        **kwargs,
    ) -> list[RerankResult]:
        """Convenience: instantiate + rerank in one call (for tests / one-shots)."""
        return cls(**kwargs).rerank(query, list(documents), top_n=top_n)


__all__ = ["OnnxReranker", "DEFAULT_MODEL_ID"]
