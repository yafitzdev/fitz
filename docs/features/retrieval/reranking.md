# ONNX Cross-Encoder Reranking

BM25 is the broad recall stage. The reranker adds a local `(query, candidate)`
relevance score before source reading and evidence compilation.

## Default Runtime

| Property | Current default |
|---|---|
| Model | `Alibaba-NLP/gte-reranker-modernbert-base` |
| Runtime | raw `onnxruntime` on CPU |
| Precision | INT8 |
| Input cap | 512 tokens per pair |
| Batch size | 1 |
| Workers | 2 concurrent forward passes |
| Pair cache | 4,096-entry process-local LRU |

The tokenizer and model load lazily on the first rerank call. Reranking never
uses the configured chat endpoint and has no governance responsibility.

## Candidate Budget

`rerank_candidates` is the moderate-query base and defaults to 32:

| Profile | Scored prefix |
|---|---:|
| Narrow | 24 |
| Moderate | 32 |
| Broad/exploratory | 48 |
| Evidence-closure pass | 16 |

The effective count is bounded by candidates actually recalled and respects
`rerank_k` and `rerank_min_addresses`. Very small pools bypass reranking.

The unscored BM25 tail is not deleted. Evidence-contract and concrete-table-row
rescue can still inspect the full recall pool, so the neural budget controls CPU
cost without redefining the lexical recall cutoff.

## Exact Deduplication And Cache

Byte-identical documents inside one call are scored once. A repeated
`(model, query, document)` pair can reuse its cached score. Cache keys are
SHA-256 digests, so the cache does not retain source text as keys.

No threshold-based eviction is used. Low-scoring pairs still occupy only the
fixed LRU budget, and removing them early would make repeated-query performance
depend on a model-score policy without reducing current-call inference cost.

## Input Handling

The model's 512-token limit is unchanged. Candidate construction uses bounded
source-faithful text; long-document handling can provide a query-centered
excerpt for ranking while delivered evidence retains the selected original
source content.

One-label sequence-classification heads use their scalar logit. Compatible
two-label heads use positive minus negative logit. Model-logit magnitudes are
not calibrated probabilities and are meaningful only for ranking within a
query.

## Measured Decision

The accepted matched 60-query SciFact hardening run measured 7.43 seconds mean
end-to-end latency, 6.77 seconds p50, and 12.56 seconds p95, with relevant
delivered evidence unchanged at 47/60. The isolated precision probe measured
INT8 reranking at 4.204 seconds mean and FP32 at 7.598 seconds mean; FP32 ranked
better in that probe but used more time and memory. INT8 remains the package
default with that quality tradeoff recorded.

See [Benchmarks](../../BENCHMARK.md) for the paired scores, intervals, hardware
limits, and interpretation.

## Configuration

```yaml
rerank: onnx
rerank_candidates: 32
rerank_k: 10
rerank_min_addresses: 2
```

An alternate compatible Hugging Face sequence-classification repository can be
selected with `rerank: onnx/<model-id>` only when it ships the expected
tokenizer and `onnx/model_int8.onnx`. Other repository layouts require direct
low-level `OnnxReranker` construction and cannot currently be selected through
the engine YAML. Compatibility, model size, and runtime performance are then
deployment responsibilities.

## Implementation

- `fitz_sage/engines/fitz_krag/retrieval/reranker.py`
- `fitz_sage/llm/providers/onnx_reranker.py`
- `fitz_sage/llm/config.py`

## Boundaries

- The scorer is pointwise; it does not optimize coverage of a result set.
- Candidate budgeting can leave some recalled items unscored.
- A 512-token pair cap is not unlimited long-document understanding.
- Cache reuse requires exact query and candidate text.

## Related

- [Sparse Search](sparse-search.md)
- [Three-Stage Strategy](three-stage-strategy.md)
- [Enterprise Retrieval Measurement](../../evaluation/enterprise-rag-bench-2026-08-01.md)
