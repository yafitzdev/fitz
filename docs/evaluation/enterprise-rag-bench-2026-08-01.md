# EnterpriseRAG-Bench Holdout (2026-08-01)

This is Fitz-Sage's first frozen, full-corpus enterprise retrieval evaluation.
It measures the real source-folder-to-evidence pipeline over more than half a
million documents from nine common company source types. It does not score
answer prose, data cleanup, or Pyrrho's governance quality.

The result is useful but mixed. The INT8 reranker produced a repeatable quality
gain. The full pipeline improved delivered nDCG@10 over literal Fitz-Sage, but
the corresponding final-ranking gain was not conclusive on the holdout. Managed
Qwen expansion broadened the search as designed, but did not improve aggregate
quality on this corpus. Multi-document project questions remain a genuine
ranking weakness, and repeated evidence-closure recall is the dominant latency
cost.

## Run Identity

- Dataset: EnterpriseRAG-Bench release `v1.0.0`
- Repository: <https://github.com/onyx-dot-app/EnterpriseRAG-Bench>
- Paper: <https://arxiv.org/abs/2605.05253>
- Holdout aggregate run: `1785549941-8fea6366`
- Development aggregate run: `1785537780-903e0efd`
- Git commit: `d6ac9799ef4a0b55058913a40d899e15ce2e5d3e`
- Worktree at every measured child-run start: clean
- Split manifest: `benchmarks/fixtures/enterprise_rag_split_v1.json`
- Canonical split digest:
  `74c2d04b613430f89052fa8a8f5fb9853e6f310289d2c9faa9381d89aaeabba5`
- Index mode: source-only; optional document enrichment was disabled
- Expansion model: `onnx-community/Qwen3-0.6B-DQ-ONNX`
- Reranker: INT8 `Alibaba-NLP/gte-reranker-modernbert-base`, batch size 1
- Governance: Fitz-Sage's exact configured Pyrrho model, with no override
- Holdout executions: 1,312 Fitz-Sage queries across four paired variants
- Bootstrap: 2,000 deterministic paired percentile samples, seed `20260731`
- Holdout wall time: 51,256.8 seconds, about 14 hours 14 minutes
- Machine: AMD Ryzen 5 9600, 6 cores / 12 threads, 47.1 GiB visible RAM,
  Windows 11 Home, Python 3.12.10
- Operational and paired-integrity gates: passed

No retrieval score was used to select a question. No ranking behavior was
changed after holdout scores were inspected.

## Corpus And Questions

The official archive contains 511,962 UTF-8 text files and 2,473,634,648
document bytes. Fitz-Sage indexed 511,961 files. The corpus adapter preserved
the original bytes and source hierarchy without rewriting, normalization,
generated metadata, summaries, or benchmark labels.

| Source | Physical documents |
|---|---:|
| Confluence | 5,189 |
| Fireflies | 10,173 |
| GitHub | 8,052 |
| Gmail | 121,390 |
| Google Drive | 25,108 |
| HubSpot | 15,017 |
| Jira | 6,120 |
| Linear | 35,308 |
| Slack | 285,605 |
| **Total** | **511,962** |

The release contains 500 questions. Of those, 470 have expected documents and
are scored; 10 high-level and 20 information-not-found questions are unscored.
The 470 scored questions contain 741 unique relevance links to 722 unique gold
document IDs. Four official document IDs are duplicated across eight physical
files. Every physical file remains indexed, while official IDs are deduplicated
only when computing relevance metrics.

One official synthetic safety document contained a literal PHP web-shell
example and was quarantined by Microsoft Defender as
`Backdoor:PHP/Perhetshell.B!dha`. It is not relevant to any benchmark question.
The file was explicitly excluded from both Fitz-Sage and plain BM25 without
changing antivirus settings or rewriting the source. The verified original
remains in the official ZIP.

Dataset integrity:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `all_documents.zip` | 1,256,181,062 | `9d1174928696ad08bc15f3f104739519de633c1605a4ec2034e0e3c0087bc5cd` |
| `questions.jsonl` | 764,927 | `f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905` |
| Adapter mapping | 140,021,786 | `34eba9f3acb864489c65a86c91806dc2b4a7d678bab91911d20927326b1588a7` |

Preparing and verifying the byte-preserving corpus projection took 621.5
seconds. The persisted Fitz-Sage database is 7,749,525,504 bytes, its source
manifest is 504,575,113 bytes, and the benchmark-local BM25 database is
4,181,131,264 bytes. These are local storage observations, not package-size
claims.

## Frozen Split

The category-stratified SHA-256 split was committed before retrieval ran. The
development set contains 142 questions and the untouched holdout contains 328.

| Category | Development | Holdout |
|---|---:|---:|
| Basic | 53 | 122 |
| Completeness | 6 | 14 |
| Conflicting information | 6 | 14 |
| Constrained | 9 | 21 |
| Intra-document reasoning | 12 | 28 |
| Miscellaneous | 6 | 14 |
| Project-related | 12 | 28 |
| Semantic | 38 | 87 |
| **Total** | **142** | **328** |

The development split was used to validate the runner and one general
reliability fix. The holdout remained score-blind until all four variants had
completed.

## Method

All variants query the same verified source index, use the same query order,
candidate budgets, evidence compiler, and exact pinned Pyrrho model.

| Variant | Managed Qwen query terms | INT8 cross-encoder |
|---|---:|---:|
| `literal` | off | off |
| `expansion` | on | off |
| `reranker` | off | on |
| `full` | on | on |

The benchmark-local reference is SQLite FTS5 BM25 with the `unicode61`
analyzer. It searches whole source documents and does not run Fitz-Sage query
planning, reranking, evidence compilation, closure, or Pyrrho. It is a quality
reference, not a latency-equivalent product path.

Primary quality is Recall@50 at broad recall. nDCG@10 measures ranking quality.
Metrics are also retained after reranking, final selection, evidence
compilation, and exact evidence delivery. Final and delivered lists normally
contain about ten documents, so their Recall@50 means recall over the complete
bounded output, not a 50-document list.

The gate is operational and integrity-only. No quality threshold was selected
after seeing scores.

## Holdout Results

| Variant | Broad Recall@50 | Recall nDCG@10 | Final Recall | Final nDCG@10 | Delivered Recall | Delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain BM25 | 0.8092 | 0.5973 | - | - | - | - | 2.49s |
| `literal` | 0.7824 | 0.5764 | 0.6569 | 0.5276 | 0.6450 | 0.5279 | 29.14s |
| `expansion` | 0.7801 | 0.5815 | 0.6611 | 0.5311 | 0.6491 | 0.5324 | 32.82s |
| `reranker` | 0.7824 | 0.5764 | 0.7006 | 0.5768 | 0.6883 | 0.5876 | 39.49s |
| `full` | 0.7801 | 0.5815 | 0.6936 | 0.5629 | 0.6816 | 0.5780 | 44.20s |

Plain BM25's mean latency excludes every model-backed and evidence-processing
stage. Its broad Recall@50 exceeded literal Fitz-Sage by 0.0269. This is a real
recall boundary on this corpus: typed and hierarchical lexical recall did not
beat direct whole-document full-text search at the 50-document cutoff.

The reranker-only variant produced the strongest final and delivered ranking.
It raised final nDCG@10 from 0.5276 to 0.5768 and delivered nDCG@10 from 0.5279
to 0.5876. The full pipeline remained better than literal delivery, but it was
not the best measured variant.

### Paired component effects

| Change | Final Recall delta | Final nDCG@10 delta | Delivered nDCG@10 delta | Mean latency delta |
|---|---:|---:|---:|---:|
| Qwen, no reranker | +0.0041 `[-0.0165, +0.0244]` | +0.0034 `[-0.0047, +0.0120]` | +0.0045 `[-0.0042, +0.0136]` | +3.68s |
| Reranker, no Qwen | +0.0437 `[+0.0057, +0.0825]` | +0.0492 `[+0.0125, +0.0875]` | +0.0597 `[+0.0238, +0.0988]` | +10.35s |
| Qwen, reranker on | -0.0071 `[-0.0243, +0.0097]` | -0.0140 `[-0.0292, +0.0013]` | -0.0096 `[-0.0249, +0.0046]` | +4.70s |
| Reranker, Qwen on | +0.0325 `[-0.0056, +0.0711]` | +0.0318 `[-0.0054, +0.0701]` | +0.0456 `[+0.0119, +0.0800]` | +11.38s |
| Full versus literal | +0.0366 `[-0.0017, +0.0785]` | +0.0352 `[-0.0029, +0.0721]` | +0.0501 `[+0.0147, +0.0851]` | +15.06s |

Intervals are paired 95% bootstrap intervals. The reranker-only quality gains
are positive. Full versus literal has a positive delivered-ranking gain, while
its final recall and final nDCG intervals cross zero. Qwen's isolated quality
effects are inconclusive.

This result does not justify disabling Qwen. Its role is to create broad
semantic-to-lexical recall where literal BM25 cannot bridge vocabulary. That
broadening is intentional, even when new candidates do not improve one
benchmark's top-k score. This corpus does not isolate synonym-only queries, and
the original literal query remains present in every expanded search.

## Query Shape

The strongest category effects were not uniform:

- Basic questions gained `+0.0693` final nDCG@10 under full versus literal,
  with a positive 95% interval `[+0.0051, +0.1391]`.
- Semantic questions gained `+0.0651` under full, but the interval crossed
  zero. Reranker-only gained `+0.0831` with a positive interval
  `[+0.0046, +0.1690]`.
- Project-related questions lost `-0.0796` under full, with a negative interval
  `[-0.1505, -0.0057]`. Reranker-only showed nearly the same loss.
- Completeness, conflict, constrained, intra-document, and miscellaneous
  category deltas were inconclusive at their small holdout sample sizes.

Absolute holdout final nDCG@10:

| Category | Queries | `literal` | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|---:|
| Basic | 122 | 0.5855 | 0.5806 | 0.6794 | 0.6548 |
| Completeness | 14 | 0.3309 | 0.3581 | 0.3665 | 0.3362 |
| Conflicting information | 14 | 0.8753 | 0.8734 | 0.8727 | 0.8508 |
| Constrained | 21 | 0.7121 | 0.7387 | 0.6549 | 0.6690 |
| Intra-document reasoning | 28 | 0.7072 | 0.7048 | 0.6910 | 0.7196 |
| Miscellaneous | 14 | 0.8929 | 0.8879 | 0.9501 | 0.9286 |
| Project-related | 28 | 0.5023 | 0.4918 | 0.4254 | 0.4227 |
| Semantic | 87 | 0.2692 | 0.2836 | 0.3523 | 0.3343 |

The project-related loss is a Fitz-Sage architecture concern, not a cleanup or
Pyrrho issue. These questions often require a coherent set of documents. A
pointwise cross-encoder can rank individually plausible documents while
reducing set coverage.

The same pattern appears when grouping by expected-document count:

| Variant | Gold cardinality | Queries | Broad Recall@50 | Final Recall | Final nDCG@10 | Delivered nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| `literal` | single | 265 | 0.7774 | 0.6604 | 0.5192 | 0.5160 |
| `reranker` | single | 265 | 0.7774 | 0.7132 | 0.5892 | 0.5952 |
| `full` | single | 265 | 0.7736 | 0.7057 | 0.5751 | 0.5832 |
| `literal` | multiple | 63 | 0.8034 | 0.6425 | 0.5633 | 0.5780 |
| `reranker` | multiple | 63 | 0.8034 | 0.6478 | 0.5247 | 0.5557 |
| `full` | multiple | 63 | 0.8076 | 0.6427 | 0.5115 | 0.5562 |

Reranking clearly helps single-document lookups. It does not improve
multi-document final recall and lowers their ranking quality. Future work
should test a set-aware coverage or diversity stage after pointwise scoring;
it should not add benchmark-specific aliases or data normalization.

## Source Types

Source groups can overlap when a question expects documents from more than one
system. The table is diagnostic and not a source leaderboard.

| Source | Questions | Literal final nDCG@10 | Reranker final nDCG@10 | Full final nDCG@10 | Full delivered nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Confluence | 76 | 0.5401 | 0.5347 | 0.5262 | 0.5696 |
| Fireflies | 17 | 0.4962 | 0.5629 | 0.5313 | 0.5350 |
| GitHub | 41 | 0.5207 | 0.5183 | 0.5152 | 0.5744 |
| Gmail | 46 | 0.5773 | 0.6311 | 0.5795 | 0.5496 |
| Google Drive | 35 | 0.4523 | 0.5804 | 0.5949 | 0.6317 |
| HubSpot | 21 | 0.3644 | 0.5884 | 0.4900 | 0.4900 |
| Jira | 76 | 0.6167 | 0.6094 | 0.5829 | 0.5977 |
| Linear | 43 | 0.5417 | 0.4925 | 0.5262 | 0.5622 |
| Slack | 56 | 0.4919 | 0.5732 | 0.5584 | 0.5438 |

No source-specific rules were added.

## Latency

| Variant | Mean | p50 | p95 | Maximum | Qwen mean | Recall mean | Rerank mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `literal` | 29.14s | 26.85s | 52.02s | 67.47s | 0.00s | 24.37s | 0.00s |
| `expansion` | 32.82s | 29.65s | 62.72s | 74.99s | 1.61s | 26.39s | 0.00s |
| `reranker` | 39.49s | 37.33s | 70.38s | 344.61s | 0.00s | 27.12s | 7.07s |
| `full` | 44.20s | 39.50s | 81.40s | 108.95s | 1.95s | 29.27s | 7.62s |

The surprising bottleneck is recall orchestration, not direct Qwen or
cross-encoder inference. Recall consumed 83.6% of literal latency and 66.2% of
full latency. The reported group includes initial recall plus repeated
evidence-closure searches. Pyrrho planning and decision averaged 2.33 seconds
in the full variant; read and context expansion were small.

One reranker-only query took 344.61 seconds. Its cross-encoder work took 7.97
seconds while repeated recall took 330.45 seconds. The same query completed in
40 to 48 seconds under the other variants, so this is an observed transient
recall/SQLite/OS outlier rather than evidence that reranking itself required
five minutes. It remains in the aggregate and is not discarded.

The next latency work should profile and bound evidence-closure recall while
preserving the broad first-stage pool. Reducing Qwen output or the reranker's
model input limit would not address the measured main cost.

## Development Result And Reliability Fix

The development split produced the same broad direction:

| Variant | Broad Recall@50 | Final nDCG@10 | Delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|
| `literal` | 0.8097 | 0.5115 | 0.5123 | 30.23s |
| `expansion` | 0.8244 | 0.5085 | 0.5065 | 34.13s |
| `reranker` | 0.8097 | 0.5587 | 0.5733 | 34.78s |
| `full` | 0.8244 | 0.5654 | 0.5861 | 39.80s |

One development query exposed malformed JSON from the optional Qwen keyword
step. Before the holdout, Fitz-Sage was changed so that this optional expansion
failure is traced and logged, then retrieval continues with the unchanged
literal query plan. It does not reinterpret the output, add heuristic terms,
or alter Pyrrho behavior. Full configured query-intelligence failures remain
visible.

On the holdout, Qwen expanded 316/328 queries, returned no terms for one, and
produced malformed output for 11 (3.35%). All 328 queries still completed.
The exact status is retained in every record. The fallback is a general
availability fix, not score-directed benchmark tuning.

## Boundaries Established

This run supports the following claims:

- Fitz-Sage can ingest and query a verified 511,961-file enterprise-shaped
  corpus without source-specific cleanup or rewriting.
- The INT8 reranker improves overall ranking and delivery quality on unseen
  questions.
- Optional semantic expansion can fail without taking literal retrieval down.
- Exact user data remains untouched, and Pyrrho remains the sole governance
  owner.

It also establishes these current limitations:

- Whole-document BM25 has 2.69 percentage points more broad Recall@50 on this
  source-only corpus.
- The full pipeline averages 44.20 seconds per query on this CPU machine.
- Repeated evidence-closure recall, not model inference, is the main latency
  cost.
- Pointwise reranking is weak on multi-document project questions.
- Managed Qwen's aggregate quality benefit is unproven on this corpus, though
  its broad semantic-recall role remains intentional.
- Semantic questions remain the weakest absolute category.

## What This Does Not Establish

- The corpus is large and enterprise-shaped, but synthetic.
- Every source is already UTF-8 text. This run does not measure PDF, Office,
  OCR, archive, or raw-log cleanup behavior.
- Source-only indexing leaves optional entity and hierarchical document
  enrichment pending. This run does not measure the value or throughput of
  those summaries.
- It does not evaluate generated answers.
- It does not test private abbreviation tables, identifier normalization, or
  domain vocabulary mappings. Those remain user-owned inputs.
- Exact Pyrrho verdicts are retained but not scored as Fitz-Sage retrieval
  quality. Governance improvements belong in Pyrrho.
- Local latency and storage observations are not an SLA.

## Decision

- Keep managed Qwen as the intentional broad semantic recall leg.
- Keep the INT8 reranker; its holdout gain generalized.
- Do not add source-specific normalization, alias heuristics, or local
  governance safeguards.
- Treat set-aware multi-document ranking and evidence-closure latency as the
  next genuine Fitz-Sage architecture targets.
- Keep this holdout frozen. Reuse it only to evaluate a declared future system
  revision, not as a development loop.

Generated query-level JSON, Markdown, stdout, stderr, and checkpoints remain
under ignored benchmark workspaces. This document is the canonical committed
summary.
