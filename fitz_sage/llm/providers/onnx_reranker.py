# fitz_sage/llm/providers/onnx_reranker.py
"""
ONNX cross-encoder reranker.

Default backbone:
[`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
— a 149M-parameter ModernBERT cross-encoder that matches 1.2B-parameter
rerankers on Hit@1.

The load path pulls the **pre-built INT8 ONNX** the model repo ships
at `onnx/model_int8.onnx` (151 MB) and runs it on raw `onnxruntime` —
no on-the-fly export, no `optimum`, no `torch`, ~2.7-3.4x faster on CPU
than FP32. This mirrors how the pyrrho governance classifier loads its
pre-quantized `model_quantized.onnx`.

A custom `model_id` must ship a pre-built ONNX; point `onnx_subfolder`
/ `onnx_file` at it. If the file can't be fetched, `_load()` raises a
clear error (there is no torch-backed export fallback by design).

Public surface — implements `RerankProvider`:

    reranker = OnnxReranker()                       # gte-reranker INT8
    reranker = OnnxReranker(model_id="BAAI/bge-reranker-base",
                            onnx_subfolder="onnx",
                            onnx_file="model_quantized.onnx")
    results = reranker.rerank(query, documents, top_n=5)
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np

from fitz_sage.llm.providers.base import RerankResult

logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "Alibaba-NLP/gte-reranker-modernbert-base"
# gte-reranker-modernbert-base ships pre-built ONNX variants under onnx/;
# model_int8.onnx is the 151 MB dynamic-INT8 quantization.
DEFAULT_ONNX_SUBFOLDER = "onnx"
DEFAULT_ONNX_FILE = "model_int8.onnx"
MAX_LENGTH = 512  # gte-modernbert-base is trained at 512; longer inputs are truncated.
DEFAULT_BATCH_SIZE = 16


class OnnxReranker:
    """Cross-encoder reranker served as INT8 ONNX on CPU.

    Args:
        model_id: HuggingFace repo id of the cross-encoder. Defaults to
            `Alibaba-NLP/gte-reranker-modernbert-base`. Any HF cross-encoder
            with a `SequenceClassification` head (num_labels=1) works.
        onnx_subfolder: Repo subfolder holding the pre-built ONNX
            (`"onnx"` for the default model; `""` if the file sits at
            the repo root, as pyrrho does).
        onnx_file: Pre-built ONNX filename to load. Defaults to the INT8
            variant.
        max_length: Tokenizer truncation cap (default 512).
        batch_size: Number of `(query, doc)` pairs per ONNX forward pass.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        onnx_subfolder: str = DEFAULT_ONNX_SUBFOLDER,
        onnx_file: str = DEFAULT_ONNX_FILE,
        max_length: int = MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_id = model_id
        self._onnx_subfolder = onnx_subfolder
        self._onnx_file = onnx_file
        self._max_length = max_length
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._tokenizer = None
        self._session = None

    def _load(self) -> None:
        if self._tokenizer is not None and self._session is not None:
            return
        with self._lock:
            if self._tokenizer is not None and self._session is not None:
                return
            import os

            # transformers is used purely as a tokenizer here — silence its
            # advisory that no DL framework (torch/TF/Flax) is installed.
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            logger.info(
                f"Loading ONNX reranker {self._model_id} "
                f"({self._onnx_subfolder or '.'}/{self._onnx_file})"
            )
            try:
                onnx_path = hf_hub_download(
                    repo_id=self._model_id,
                    filename=self._onnx_file,
                    subfolder=self._onnx_subfolder or None,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not fetch the pre-built ONNX "
                    f"'{self._onnx_subfolder or '.'}/{self._onnx_file}' from "
                    f"{self._model_id}: {e}. Point `onnx_subfolder` / "
                    f"`onnx_file` at a cross-encoder ONNX the repo actually "
                    f"ships."
                ) from e
            self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

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
        n = len(documents)
        scores = [0.0] * n
        input_names = [i.name for i in self._session.get_inputs()]  # type: ignore[union-attr]
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
            # Feed only the inputs the ONNX graph declares.
            feed = {name: enc[name] for name in input_names}
            logits = self._session.run(None, feed)[0]  # type: ignore[union-attr]
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
