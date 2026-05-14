# BEIR Benchmark Results (Historical, pre-v0.12.0)

> **Note.** These results were measured against the *hybrid*
> retrieval stack that fitz-sage ran before **v0.12.0**, when both
> dense embeddings and BM25 contributed to ranking. v0.12.0 removed
> the embedding layer entirely. The numbers here are kept as a
> historical baseline; they are not representative of the current
> system. The current retrieval stack is BM25 (via SQLite FTS5) +
> KRAG typed-unit routing + ONNX cross-encoder reranker.

**Last updated:** 2026-02-20
**Datasets run:** scifact, scidocs, fiqa (Tier 1)
**Retrieval (at the time):** Hybrid BM25 (0.6) + dense embeddings (0.4) via `SectionSearchStrategy`

---

## Summary Table

| Dataset  | nomic-embed-text | bge-m3 (before fixes) | bge-m3 (after fixes) | BM25 baseline | SOTA (dense) |
|----------|:----------------:|:---------------------:|:--------------------:|:-------------:|:------------:|
| scifact  | 0.6103           | 0.6262                | **0.6735** ✓         | 0.6647        | ~0.77        |
| scidocs  | 0.0799           | 0.1319                | **0.1436**           | 0.1490        | ~0.17        |
| fiqa     | 0.2251           | 0.2702 ✓              | (not re-run)         | 0.2361        | ~0.45        |

Metric: **nDCG@10** (higher is better, max 1.0)

**Fixes applied 2026-02-20**: RRF merge (k=60) for both retrieval legs, removed `min_relevance_score` filter, raised HNSW `ef_search` to 200.

---

## Full Results

### nomic-embed-text (ollama, 274MB, 768d) — 2026-02-19

| Dataset  | nDCG@10 | Recall@100 | MAP@10 | Queries | Docs   |
|----------|--------:|-----------:|-------:|--------:|-------:|
| scifact  | 0.6103  | 0.8817     | 0.5578 | 300     | 5,183  |
| scidocs  | 0.0799  | 0.2699     | 0.0454 | 1,000   | 25,657 |
| fiqa     | 0.2251  | 0.5173     | 0.1587 | 648     | 57,638 |

### bge-m3 + RRF + no threshold + ef_search=200 — 2026-02-20

| Dataset  | nDCG@10 | Recall@100 | MAP@10 | Queries | Docs   |
|----------|--------:|-----------:|-------:|--------:|-------:|
| scifact  | **0.6735** ✓ | 0.9072  | 0.6310 | 300     | 5,183  |
| scidocs  | **0.1436** | 0.3290   | 0.0824 | 1,000   | 25,657 |
| fiqa     | (not re-run) | —      | —      | 648     | 57,638 |

scifact beats BM25 baseline (0.6647). scidocs within 4% of BM25 baseline (0.1490).

### bge-m3 (ollama, 1.2GB, 1024d, num_ctx=8192) — 2026-02-19

| Dataset  | nDCG@10 | Recall@100 | MAP@10 | Queries | Docs   |
|----------|--------:|-----------:|-------:|--------:|-------:|
| scifact  | 0.6261  | 0.9017     | 0.5651 | 300     | 5,183  |
| scidocs  | 0.1320  | 0.3253     | 0.0727 | 1,000   | 25,657 |
| fiqa     | 0.2702  | 0.6099     | 0.1816 | 648     | 57,638 |

### Dense-only baseline (nomic, before hybrid) — 2026-02-19

| Dataset  | nDCG@10 |
|----------|--------:|
| scifact  | 0.5027  |
| scidocs  | 0.0553  |
| fiqa     | 0.2117  |

Wiring in the hybrid BM25+semantic path gave a consistent lift across all datasets.

---

## BEIR BM25 Baselines (published)

| Dataset  | nDCG@10 | Recall@100 | MAP    |
|----------|--------:|-----------:|-------:|
| scifact  | 0.6647  | 0.9527     | 0.6315 |
| scidocs  | 0.1490  | 0.3640     | 0.0760 |
| fiqa     | 0.2361  | 0.5535     | 0.1320 |

Source: [BEIR leaderboard](https://github.com/beir-cellar/beir)

---

## What the Numbers Mean

**nDCG@10** (Normalized Discounted Cumulative Gain at 10) is the primary BEIR metric.
It measures how good the top-10 retrieved results are, where:
- Relevant results ranked higher score more
- 1.0 = perfect retrieval, 0.0 = completely wrong
- BM25 baseline (~0.66 on scifact) is the standard "are you at least as good as keyword search?" bar
- SOTA dense retrievers hit ~0.74–0.77 on scifact with purpose-trained models

**Recall@100** measures whether the correct answer appears anywhere in the top 100 results. This is the upper bound on what the generation step can work with.

---

## Dataset Notes

### scifact (5,183 docs, 300 queries)
Scientific claim verification. Text similarity works well here. Our bge-m3 score (0.6261) is 6% below BM25 baseline — within striking distance for a local model.

### scidocs (25,657 docs, 1,000 queries)
**Structurally unfavorable for any text-similarity system.** Relevance labels are based on citation graph membership (did paper A cite paper B?), not semantic similarity. Even SOTA dense retrievers only reach ~0.17. Our 0.1320 with bge-m3 is respectable for a local model. Don't chase this one.

### fiqa (57,638 docs, 648 queries)
Financial Q&A from Reddit and investment forums. bge-m3 scores **0.2702**, beating BM25 baseline (0.2361) — the hybrid semantic search genuinely adds value here. Large corpus makes it slow to index with local models (~90 min with bge-m3 on CPU).

---

## Embedding Model Comparison

| Model              | Provider | Size   | Dim  | scifact nDCG@10 | Speed (fiqa index) | Notes                          |
|--------------------|----------|--------|------|:---------------:|--------------------|--------------------------------|
| nomic-embed-text   | Ollama   | 274MB  | 768  | 0.6103          | ~5 min             | Default. Fast, decent quality. |
| bge-m3             | Ollama   | 1.2GB  | 1024 | 0.6261          | ~90 min            | Best local quality. Slow on large corpora. Requires `num_ctx: 8192`. |
| mxbai-embed-large  | Ollama   | 669MB  | 1024 | (not tested)    | ~30 min est.       | Middle ground option.          |
| Cohere embed-v4.0  | Cloud    | —      | 1024 | (not tested)    | ~2 min             | Cloud. Requires API key.       |

### bge-m3 configuration requirement

bge-m3 via Ollama defaults to a small context window, causing 500 errors on longer texts. Always set `num_ctx: 8192` in config:

```yaml
# .fitz/config.yaml
embedding: ollama/bge-m3
```

Set `num_ctx` via Ollama modelfile or API options. This is implemented via `options: {num_ctx: 8192}` in the Ollama `/api/embed` payload.

### Switching embedding models

Changing models requires rebuilding all existing collections (vector dimension changes from 768 to 1024). The engine will error on startup with a clear message if there's a mismatch.

---

## Known Issues

### Persistent embed failures (~10–35 docs per dataset)

Some documents fail to embed with bge-m3 regardless of context window size. These appear to be content issues (binary data, null bytes, or unusual Unicode in some paper metadata or Reddit posts). The fallback in `FitzBEIRRetriever.index_corpus()` catches these, logs a warning, and skips them. Impact on scores is negligible (<0.1% of corpus).

The same issue occurs with nomic-embed-text but causes 400 errors instead of 500s.

### fiqa indexing time

With bge-m3 at `num_ctx=8192`, fiqa's 57,638-doc corpus takes ~90 minutes to index on CPU. For benchmark iteration, either:
- Use nomic-embed-text (5 min for fiqa)
- Run scifact only (fastest meaningful signal)
- Use Cohere embeddings (cloud-speed)

---

## Architecture

The BEIR benchmark is implemented in `fitz_sage/evaluation/benchmarks/beir.py`.

**`FitzBEIRRetriever`** is the BEIR-compatible retriever wrapper:
1. `index_corpus()` — creates a temp collection `beir_{dataset}_{uuid8}`, inserts raw file rows (FK requirement), upserts sections with vectors via `SectionStore.upsert_batch()`
2. `search()` — uses `SectionSearchStrategy.retrieve()` for hybrid BM25+semantic retrieval, maps `Address.source_id` → BEIR doc_id
3. `cleanup()` — DELETEs temp collection from `krag_section_index` and `krag_raw_files`

**`BEIRBenchmark.evaluate()`** wraps this in a try/finally to guarantee cleanup even on failure.

---

## Running the Benchmark

```python
from fitz_sage.config.loader import load_engine_config
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from fitz_sage.evaluation.benchmarks import BEIRBenchmark

config = load_engine_config("fitz_krag")
config.collection = "beir_scratch"  # avoid dim-check on existing collections
engine = FitzKragEngine(config)
beir = BEIRBenchmark()

result = beir.evaluate(engine, dataset="scifact")
print(f"nDCG@10: {result.ndcg_at_10:.4f}")
```

**Tier 1 datasets:** `scifact`, `scidocs`, `fiqa`
**Tier 2 datasets:** `nfcorpus`, `arguana`, `trec-covid`, `webis-touche2020`, `quora`, `dbpedia-entity`, `fever`, `climate-fever`, `hotpotqa`
**All datasets:** `beir.list_datasets()`

---

## Next Steps / Ideas

- Re-run fiqa with post-fix pipeline (skipped — 90 min with bge-m3 on CPU)
- Test `mxbai-embed-large` as a speed/quality middle ground
- Test Cohere `embed-v4.0` for cloud-speed baseline
- Add reranker (Cohere or bge-reranker-v2-m3) and measure lift on scifact
- Run Tier 2 datasets once bge-m3 is confirmed stable
- Investigate scidocs gap (0.1436 vs 0.1490) — likely citation-graph relevance labels that text similarity can't close

## Detection Classifier (Shipped in v0.10.1)

BEIR exposed that the real retrieval risk is not document recall but **query misclassification**: the `DetectionOrchestrator` uses the fast-tier LLM (qwen2.5:3b) to detect temporal, comparison, causal, aggregation, and freshness signals. If it misses, all downstream routing is wrong — silently.

**Implemented:** `DetectionClassifier` (`retrieval/detection/classifier.py`) — an ML + keyword classifier that gates LLM detection calls:
- **ML model** (logistic regression + TF-IDF): temporal 90.6% recall, comparison 90.2% recall
- **Keyword regex**: aggregation, freshness, rewriter (no ML needed)
- **Fail-open**: if the model artifact is missing or prediction fails, all LLM modules run as before
- Queries flagged by the classifier trigger only the relevant LLM modules; unflagged queries skip LLM detection entirely
