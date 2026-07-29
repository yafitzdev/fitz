# fitz_sage/llm/providers/onnx_reranker.py
"""
ONNX cross-encoder reranker.

Default backbone:
[`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
— a 149M-parameter ModernBERT cross-encoder that matches 1.2B-parameter
rerankers on Hit@1.

The lazy load + forward pass come from `OnnxEncoderBackend`, the shared
encoder backend: it pulls the **pre-built INT8 ONNX** the model repo
ships at `onnx/model_int8.onnx` (151 MB) and runs it on raw
`onnxruntime` — no on-the-fly export, no `optimum`, no `torch`,
~2.7-3.4x faster on CPU than FP32.

A custom `model_id` must ship a pre-built ONNX; point `onnx_subfolder`
/ `onnx_file` at it. If the file can't be fetched, the loader raises a
clear error (there is no torch-backed export fallback by design).

Public surface — implements `RerankProvider`:

    reranker = OnnxReranker()                       # gte-reranker INT8
    reranker = OnnxReranker(model_id="BAAI/bge-reranker-base",
                            onnx_subfolder="onnx",
                            onnx_file="model_quantized.onnx")
    results = reranker.rerank(query, documents, top_n=5)
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from fitz_sage.encoders.onnx import OnnxEncoderBackend
from fitz_sage.llm.providers.base import RerankResult

DEFAULT_MODEL_ID = "Alibaba-NLP/gte-reranker-modernbert-base"
# gte-reranker-modernbert-base ships pre-built ONNX variants under onnx/;
# model_int8.onnx is the 151 MB dynamic-INT8 quantization.
DEFAULT_ONNX_SUBFOLDER = "onnx"
DEFAULT_ONNX_FILE = "model_int8.onnx"
MAX_LENGTH = 512  # gte-modernbert-base is trained at 512; longer inputs are truncated.
DEFAULT_BATCH_SIZE = 1
DEFAULT_WORKERS = 2
DEFAULT_CACHE_SIZE = 4096


class OnnxReranker(OnnxEncoderBackend):
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
        workers: Maximum concurrent ONNX forward passes.
        cache_size: Maximum exact `(model, query, document)` scores retained
            in memory. Set to zero to disable the cache.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        onnx_subfolder: str = DEFAULT_ONNX_SUBFOLDER,
        onnx_file: str = DEFAULT_ONNX_FILE,
        max_length: int = MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        workers: int = DEFAULT_WORKERS,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")

        super().__init__(model_id=model_id, onnx_file=onnx_file, onnx_subfolder=onnx_subfolder)
        self._max_length = max_length
        self._batch_size = batch_size
        self._workers = workers
        self._cache_size = cache_size
        self._score_cache: OrderedDict[bytes, float] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_namespace = "\0".join(
            (model_id, onnx_subfolder, onnx_file, str(max_length))
        ).encode("utf-8")
        self.last_trace: dict[str, Any] = {}

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score `(query, doc)` pairs and sort by score with stable ties."""
        if not documents:
            self.last_trace = self._shortcut_trace("no_documents", 0)
            return []
        if len(documents) == 1:
            self.last_trace = self._shortcut_trace("single_document", 1)
            return [RerankResult(index=0, score=1.0)]

        scores = self._score_pairs(query, documents)

        results = [RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)]
        results.sort(key=lambda r: (-r.score, r.index))

        if top_n is not None:
            results = results[:top_n]
        return results

    def _shortcut_trace(self, shortcut: str, document_count: int) -> dict[str, Any]:
        return {
            "requested_document_count": document_count,
            "unique_document_count": document_count,
            "duplicate_document_count": 0,
            "cache_hit_count": 0,
            "scored_document_count": 0,
            "forward_pass_count": 0,
            "batch_size": self._batch_size,
            "workers": self._workers,
            "cache_size": self._cache_size,
            "shortcut": shortcut,
        }

    def _score_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Score exact unique pairs, reusing bounded process-local results."""
        unique_documents = list(dict.fromkeys(documents))
        score_by_document: dict[str, float] = {}
        cache_keys: dict[str, bytes] = {}
        uncached_documents: list[str] = []
        cache_hits = 0

        for document in unique_documents:
            key = self._cache_key(query, document)
            cache_keys[document] = key
            cached = self._cache_get(key)
            if cached is None:
                uncached_documents.append(document)
                continue
            score_by_document[document] = cached
            cache_hits += 1

        uncached_scores = self._score_uncached_pairs(query, uncached_documents)
        for document, score in zip(uncached_documents, uncached_scores, strict=True):
            score_by_document[document] = score
            self._cache_put(cache_keys[document], score)

        self.last_trace = {
            "requested_document_count": len(documents),
            "unique_document_count": len(unique_documents),
            "duplicate_document_count": len(documents) - len(unique_documents),
            "cache_hit_count": cache_hits,
            "scored_document_count": len(uncached_documents),
            "forward_pass_count": _batch_count(len(uncached_documents), self._batch_size),
            "batch_size": self._batch_size,
            "workers": self._workers,
            "cache_size": self._cache_size,
        }
        return [score_by_document[document] for document in documents]

    def _score_uncached_pairs(self, query: str, documents: list[str]) -> list[float]:
        """Run bounded batches while keeping at most ``workers`` jobs pending."""
        if not documents:
            return []

        batches = [
            documents[start : start + self._batch_size]
            for start in range(0, len(documents), self._batch_size)
        ]
        if self._workers == 1 or len(batches) == 1:
            scores: list[float] = []
            for batch in batches:
                scores.extend(self._score_batch(query, batch))
            return scores

        scores = []
        pending: deque[tuple[int, Future[Any]]] = deque()
        with ThreadPoolExecutor(
            max_workers=min(self._workers, len(batches)),
            thread_name_prefix="fitz-rerank",
        ) as executor:
            for batch in batches:
                encoded = self._encode_batch(query, batch)
                pending.append((len(batch), executor.submit(self._run, encoded)))
                if len(pending) >= self._workers:
                    scores.extend(self._resolve_batch(pending.popleft()))
            while pending:
                scores.extend(self._resolve_batch(pending.popleft()))
        return scores

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        scores = _scores_from_logits(self._run(self._encode_batch(query, documents)))
        self._validate_score_count(scores, len(documents))
        return scores

    def _encode_batch(self, query: str, documents: list[str]) -> Any:
        return self._encode(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )

    @staticmethod
    def _resolve_batch(pending: tuple[int, Future[Any]]) -> list[float]:
        expected_count, future = pending
        scores = _scores_from_logits(future.result())
        OnnxReranker._validate_score_count(scores, expected_count)
        return scores

    @staticmethod
    def _validate_score_count(scores: list[float], expected_count: int) -> None:
        if len(scores) != expected_count:
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {expected_count} documents"
            )

    def _cache_key(self, query: str, document: str) -> bytes:
        digest = hashlib.sha256()
        for value in (self._cache_namespace, query.encode("utf-8"), document.encode("utf-8")):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        return digest.digest()

    def _cache_get(self, key: bytes) -> float | None:
        if self._cache_size == 0:
            return None
        with self._cache_lock:
            score = self._score_cache.get(key)
            if score is not None:
                self._score_cache.move_to_end(key)
            return score

    def _cache_put(self, key: bytes, score: float) -> None:
        if self._cache_size == 0:
            return
        with self._cache_lock:
            self._score_cache[key] = score
            self._score_cache.move_to_end(key)
            while len(self._score_cache) > self._cache_size:
                self._score_cache.popitem(last=False)


def _scores_from_logits(logits: Any) -> list[float]:
    """Normalize one- and two-label sequence-classification logits."""
    import numpy as np

    arr = np.asarray(logits)
    if arr.ndim == 2 and arr.shape[1] == 1:
        batch_scores = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] == 2:
        batch_scores = arr[:, 1] - arr[:, 0]
    else:
        batch_scores = arr.reshape(arr.shape[0], -1)[:, 0]
    return [float(score) for score in batch_scores.tolist()]


def _batch_count(item_count: int, batch_size: int) -> int:
    return (item_count + batch_size - 1) // batch_size


__all__ = ["OnnxReranker", "DEFAULT_MODEL_ID"]
