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

After FTS5 / BM25 returns a candidate set, a single INT8 ONNX
cross-encoder scores each `(query, candidate)` pair in one batched
forward pass. Re-order by the model's score, keep the top-K.

```
Query: "What's the battery warranty?"
            │
            ▼
   FTS5 + bm25() — recall
   returns ~20 candidates
            │
            ▼
   ONNX cross-encoder — precision           ◀── standard product path
   one forward pass over (q, doc) pairs
   ~30–100 ms CPU for 10–20 candidates
            │
            ▼
   Top-K truly-relevant candidates
            │
            ▼
   Pyrrho governance cutoff
```

**Default backbone:** [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) —
149M-parameter ModernBERT cross-encoder. Matches 1.2B-parameter
rerankers on Hit@1, and INT8 ONNX quantisation gives 2.7–3.4× CPU
speedup with ~98% of full-precision quality.

Override via `rerank: onnx/<hf-model-id>` — e.g.
`onnx/BAAI/bge-reranker-base` for a multilingual alternative,
`onnx/jinaai/jina-reranker-v3` for higher quality at larger size.

## Why a cross-encoder, not the chat model?

Before v0.13.0, fitz-sage served reranking via a chat-completion call
(`LLMReranker`) — the model was asked to grade each candidate. That
worked but cost one chat call per query on the hot path. v0.13.0
swapped in a dedicated cross-encoder for three reasons:

1. **Latency.** ~30–100 ms CPU for 10–20 candidates vs ~500–2000 ms
   for a 7B chat model.
2. **No external dependency.** Inference is local — both the
   governance and reranking paths went chat-free in v0.13.0.
3. **Stronger ranking signal.** Cross-encoders are the textbook
   solution for `(query, doc)` relevance scoring and saturate the
   benchmark — chat-based reranking was reinventing this with worse
   inductive bias.

The same model family as the
[Pyrrho governance classifier](https://huggingface.co/yafitzdev/pyrrho-nano-g3.1):
ModernBERT-base, local CPU inference, lazy-loaded on first call, and cached for
the process lifetime. The reranker remains ONNX; Pyrrho g3.1 is a multitask
safetensors checkpoint.

## How it works

```python
# Each batch of (query, doc) pairs goes through one ONNX forward pass.
enc = tokenizer([query] * len(docs), docs,
                padding=True, truncation=True, max_length=512,
                return_tensors="np")
logits = model(**enc).logits      # shape (B, 1) for sequence-classification heads
scores = logits[:, 0]             # higher = more relevant
```

Sequence-classification head with `num_labels=1` is the standard
cross-encoder shape; 2-class heads (some BGE variants) are handled by
taking `pos_logit - neg_logit`.

### Smart skip

If the candidate pool is small (below `rerank_min_addresses`), the
reranker step is bypassed — there's nothing meaningful to rank. Same
behaviour as v0.13.x; the threshold is unchanged.

### VIP preservation

Artifact rows (architecture narrative, dependency summary) carry a
sentinel `score = 1.0` and bypass the reranker. They're "always
include" by design.

## Key design decisions

1. **Standard rerank stage.** `rerank: onnx` is the default product path,
   and the engine config does not expose a normal "rerank off" mode.
2. **Shared with pyrrho.** Both encoders subclass `OnnxEncoderBackend`
   — one `onnxruntime` + `transformers` load path, no separate
   infrastructure.
3. **Override via spec.** `rerank: onnx/<hf-model-id>` lets users
   swap in any HF cross-encoder with a `SequenceClassification` head.
4. **Lazy load.** Tokenizer + model load on first `rerank()` call,
   not at engine init — keeps startup fast.
5. **Batched forward.** Pairs go through in batches of 16 by default;
   `batch_size` is configurable per `OnnxReranker` instance.

## Configuration

### Enable (default)

```yaml
rerank: onnx        # uses Alibaba-NLP/gte-reranker-modernbert-base
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
| Sparse search (FTS5)   | Runs *before* reranking; produces the candidate pool               |
| Query expansion        | Runs *before* reranking; all expanded results land in one pool     |
| KRAG routing           | Cross-encoder sees the rewritten query, not the raw user text      |
| Multi-hop              | Reranker runs inside each hop independently                        |
| Governance (pyrrho)    | Reranker output feeds the pyrrho classifier; reranker doesn't see  |
|                        | governance decisions                                                |

## Related

- [Sparse Search (FTS5 + bm25)](sparse-search.md) — the recall layer
- [Multi-Hop Reasoning](multi-hop-reasoning.md) — reranker runs inside each hop
- [Unified Storage](../platform/unified-storage.md) — SQLite + FTS5 layer
- [Epistemic Governance (pyrrho)](../../CONSTRAINTS.md) — the next encoder in the pipeline
