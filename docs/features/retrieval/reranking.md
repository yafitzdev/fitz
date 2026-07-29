# Reranking (ONNX cross-encoder)

## Problem

BM25 + FTS5 optimises for **recall** — find candidates that contain
the right tokens. Token overlap isn't the same as true relevance:

- Two sections can both mention the query terms while only one
  actually answers the question.
- The top-5 by `bm25()` aren't always the best 5 for the user's
  *intent*.
- BM25 is intent-blind: it can't tell a "how do I do X?" question
  from a "what is X?" question.

## Solution: ONNX cross-encoder reranker

After FTS5 / BM25 returns a candidate set, an INT8 ONNX cross-encoder
scores a bounded prefix of `(query, candidate)` pairs. Fitz-Sage keeps
the full BM25 pool available to evidence-contract and concrete-row rescue
logic, but avoids paying neural inference cost for every recalled item.
The scored prefix is reordered by model score and reduced to the top-K.

```
Query: "What's the battery warranty?"
            │
            ▼
   FTS5 + bm25() — recall
   returns the full recall pool
            │
            ▼
   ONNX cross-encoder — precision           ◀── standard product path
   scores 24 / 32 / 48 candidates
   two concurrent INT8 batch-one passes
            │
            ▼
   Top-K truly-relevant candidates
            │
            ▼
   fixed evidence delivery, then one Pyrrho decision
```

**Default backbone:** [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) —
149M-parameter ModernBERT cross-encoder. Matches 1.2B-parameter
rerankers on Hit@1, and INT8 ONNX quantisation gives 2.7–3.4× CPU
speedup with ~98% of full-precision quality.

Override via `rerank: onnx/<hf-model-id>` — e.g.
`onnx/BAAI/bge-reranker-base` for a multilingual alternative,
`onnx/jinaai/jina-reranker-v3` for higher quality at larger size.

## Why a cross-encoder, not the chat model?

A dedicated cross-encoder is the right tool for `(query, document)` relevance
scoring:

1. **Bounded latency.** Cross-encoder cost is capped independently from
   the full lexical recall pool.
2. **No external dependency.** Inference is local and does not call the
   configured chat endpoint.
3. **Stronger ranking signal.** Cross-encoders are the textbook
   solution for pairwise relevance scoring.

The reranker uses local CPU inference, lazy loading, and a bounded
process-lifetime cache. Pyrrho is a separate package and remains the sole
owner of governance.

## How it works

```python
# Each uncached pair is tokenized with the model's full 512-token cap.
enc = tokenizer([query], [document],
                padding=True, truncation=True, max_length=512,
                return_tensors="np")
logits = model(**enc).logits
score = logits[0, 0]              # higher = more relevant
```

Sequence-classification head with `num_labels=1` is the standard
cross-encoder shape; 2-class heads (some BGE variants) are handled by
taking `pos_logit - neg_logit`.

The default runtime keeps two ONNX forward passes in flight. Exact duplicate
documents within a request are scored once, and exact repeated
`(model, query, document)` pairs reuse a bounded 4,096-entry LRU cache. Cache
keys are SHA-256 digests, so source text is not retained in the cache.

### Candidate budget

`rerank_candidates` is the moderate-query base budget and defaults to 32.
The deterministic retrieval profile derives:

| Query profile | Cross-encoder candidates |
|---|---:|
| Narrow | 24 |
| Moderate | 32 |
| Broad / exploratory | 48 |
| Evidence closure | 16 |

`rerank_k` and `rerank_min_addresses` remain hard lower bounds. The budget is
also limited by the number of candidates actually recalled.

This does not truncate BM25 recall. Required-modality ordering, concrete-row
preservation, and broad-corpus rescue still inspect the complete recall pool.

### Small-pool skip

If the candidate pool is small (below `rerank_min_addresses`), the
reranker step is bypassed — there's nothing meaningful to rank.

### VIP preservation

Artifact rows (architecture narrative, dependency summary) carry a
sentinel `score = 1.0` and bypass the reranker. They're "always
include" by design.

## Key design decisions

1. **Standard rerank stage.** `rerank: onnx` is the default product path,
   and the engine config does not expose a normal "rerank off" mode.
2. **Fitz-Sage-owned runtime.** The reranker subclasses
   `OnnxEncoderBackend` and uses Fitz-Sage's ONNX inference machinery.
   Pyrrho is a separate package with its own runtime.
3. **Override via spec.** `rerank: onnx/<hf-model-id>` lets users
   swap in any HF cross-encoder with a `SequenceClassification` head.
4. **Lazy load.** Tokenizer + model load on first `rerank()` call,
   not at engine init — keeps startup fast.
5. **Batch-one execution.** The INT8 default scores one pair per forward
   pass with two workers. This measured faster than larger dynamic batches
   for the shipped model on the benchmark machine.
6. **Exact reuse only.** Deduplication and caching require byte-identical
   query and document text. Fitz-Sage does not normalize identifiers or
   infer that differently formatted source strings are equivalent.
7. **Full input length.** The tokenizer cap remains 512 tokens. Candidate
   budgeting reduces pair count, not per-pair context.

## Configuration

### Enable (default)

```yaml
rerank: onnx        # uses Alibaba-NLP/gte-reranker-modernbert-base
rerank_candidates: 32
rerank_k: 10
```

### Use a different cross-encoder

```yaml
rerank: onnx/BAAI/bge-reranker-base
# rerank: onnx/jinaai/jina-reranker-v3
# rerank: onnx/cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Files

| Component                  | Path                                                              |
| -------------------------- | ----------------------------------------------------------------- |
| Pipeline step              | `fitz_sage/engines/fitz_krag/retrieval/reranker.py`               |
| ONNX reranker provider     | `fitz_sage/llm/providers/onnx_reranker.py`                        |
| Factory dispatch           | `fitz_sage/llm/config.py` (`create_rerank_provider`)              |

## Example

**Query:** "What's the warranty period for the battery?"

**After FTS5 + bm25 (top 5 by BM25):**

1. "Battery specifications: 75 kWh capacity ..."  (bm25 rank 1)
2. "Warranty terms vary by component ..."        (bm25 rank 2)
3. "The battery uses lithium-ion cells ..."      (bm25 rank 3)
4. "Battery warranty: 8 years or 100,000 miles." (bm25 rank 4)
5. "Charging the battery takes 45 minutes ..."   (bm25 rank 5)

**After ONNX cross-encoder reranker:**

1. "Battery warranty: 8 years or 100,000 miles." (rerank 8.72)
2. "Warranty terms vary by component ..."        (rerank 4.91)
3. "Battery specifications: 75 kWh capacity ..." (rerank 1.34)
4. "The battery uses lithium-ion cells ..."      (rerank 0.62)
5. "Charging the battery takes 45 minutes ..."   (rerank -0.18)

The reranker promotes the warranty-specific row over the broader
battery-spec candidates. Raw logits — magnitudes vary by backbone.

## Interaction with other features

| Feature                | Relationship                                                       |
| ---------------------- | ------------------------------------------------------------------ |
| Sparse search (FTS5)   | Produces the full pool; only a bounded prefix is model-scored       |
| Query expansion        | Runs *before* reranking; all expanded results land in one pool     |
| KRAG routing           | Cross-encoder sees the rewritten query, not the raw user text      |
| Evidence closure       | Follow-up retrieval uses the same reranking path before compilation |
| Governance (pyrrho)    | Reranker output feeds the pyrrho classifier; reranker doesn't see  |
|                        | governance decisions                                                |

## Related

- [Sparse Search (FTS5 + bm25)](sparse-search.md) — the recall layer
- [Retrieval Pipeline](../../RETRIEVAL_PIPELINE.md) — reranking and bounded evidence closure
- [Unified Storage](../platform/unified-storage.md) — SQLite + FTS5 layer
- [Epistemic Governance (pyrrho)](../../CONSTRAINTS.md) — the next encoder in the pipeline
