# Fitz-Sage Benchmark Snapshot

> Temporary working record for the README update planned after the benchmark
> program is complete. Last consolidated: 2026-07-31.

This file collects the latest accepted measurements and the decision-driving
diagnostics produced during the post-v0.15.0 hardening work. It intentionally
excludes smoke runs, interrupted runs, and superseded intermediate reports.
The measurements were made at different Git revisions, so this is not one
single release-candidate scorecard.

## Reading Rules

- Retrieval, fixed evidence delivery, query-shape recognition, and Pyrrho
  decisions are separate measurements. The Pyrrho figures must not be
  described as Fitz-Sage retrieval quality.
- `source-only` means parsed source is query-ready while optional document
  enrichment remains pending. NapierOne ingestion timings exclude Qwen
  enrichment.
- BEIR scores compare systems only within the stated dataset and run. They are
  not an official leaderboard submission.
- Timings are observations from the local six-core benchmark machine, not an
  SLA. A complete hardware fingerprint was not retained in the committed
  summaries.
- Qwen candidates competing with literal candidates inside a bounded pool is
  intentional broad-recall behavior. A fixed-cutoff regression does not by
  itself identify candidate competition as a package defect.
- User-owned data cleanup remains outside the package contract, including OCR,
  raw-log compression, private vocabulary mappings, and identifier
  normalization.

## At A Glance

| Area | Measured scale | Headline result |
|---|---:|---|
| Required production matrix | 252 capability contracts | 246/252 (97.6%) package capability; gate passed |
| Query-shape suite | 60 cases | 60/60 (100%) |
| Limitation suite | 60 cases | 52/52 asserted retrieval and delivery; 35/60 complete with accepted Pyrrho |
| Local source indexing | 18 core / 93 mixed files | 60.8 / 51.6 files/s |
| NapierOne source indexing | up to 5,005 real files | 7.27 files/s at scale; recovery gate passed |
| Broad BEIR ablation | 66,454 docs, 1,271 queries | full macro final nDCG@10 0.4365 |
| Semantic BEIR holdout | 531,605 docs, 240 queries | full macro final nDCG@10 0.6586 |
| Reranker hardening | matched 60-query SciFact sample | mean query latency 20.84s to 7.43s |

## Internal Production Matrix

The accepted local matrix was run on 2026-07-27 from fresh fixture folders and
isolated workspaces.

| Metric | Result |
|---|---:|
| Required compiled retrieval | 186/192 (96.9%) |
| Required fixed evidence delivery | 186/192 (96.9%) |
| Query-shape recognition | 60/60 (100%) |
| Combined package capability | 246/252 (97.6%) |
| Full contract including diagnostic Pyrrho modes | 193/252 (76.6%) |
| Core retrieval after adding 80 near-neighbor documents | 20/20 |
| Reload stability | 100% retrieval, delivery, and mode identity |
| Required-suite ingestion | 209/209 files |
| Production gate | pass |

Suite-level results:

| Suite | Retrieval or shape | Delivery | Purpose |
|---|---:|---:|---|
| Core | 20/20 | 20/20 | baseline behavior |
| Holdout | 47/50 | 47/50 | first unseen corpus |
| Holdout 2 | 47/50 | 47/50 | second unseen corpus |
| Core plus 80 noise documents | 20/20 | 20/20 | near-neighbor robustness |
| Query shapes | 60/60 | n/a | temporal, comparison, aggregation, narrow |
| PDF/DOCX/PPTX | 24/24 | 24/24 | rich-document facts |
| SQL/Go/Java/TypeScript/PPTX | 17/17 | 17/17 | code and base formats |
| XLSX, optional parser | 5/5 | 5/5 | optional parser path |
| Hardened boundaries | 11/11 | 11/11 | long-document, bridge, precision, structured cases |
| Limitations, non-gating | 52/52 | 52/52 | cases with explicit evidence assertions |

The six remaining required-suite package misses are one grouped code-constant
case, two coordinated-prose second clauses, one table superlative, one missing
companion service row, and one mixed table/code scheduler expression. They
remain in the suite as visible boundaries rather than case-specific tuning
targets.

### Intentional Limitation Suite

- 60 total cases were run.
- All 52 cases with evidence assertions passed retrieval and fixed delivery.
- Required recall was 100%, and no forbidden evidence was returned.
- 35/60 complete contracts passed.
- All 25 complete-contract failures were attributed to exact Pyrrho verdicts
  or failure modes while retrieval and delivery still passed.
- The run used `pyrrho-v2-nano-g1` at its current 2,048-token contract.
- Pyrrho training included benchmark-derived deterministic rows, so 35/60 is
  integration evidence, not an independent model-quality result.

### Local Fixture Performance

| Run | Discovered files | Query-ready time | Throughput | Source-index failures | Other |
|---|---:|---:|---:|---:|---|
| Core, median of 3 cold runs | 18 | 0.296s | 60.8 files/s | 0 | source-only |
| Mixed formats | 93 | 1.803s | 51.6 files/s | 0 | 1 XLSX unsupported by default CPU parser |
| Unchanged core re-point | 18 | 0.02-0.03s | n/a | 0 | counts unchanged |

The source-only core run passed 20/20 retrieval, delivery, and package
capability contracts, with 100% required recall and no forbidden evidence. It
passed 14/20 full contracts; the six failures were accepted Pyrrho outputs.

Historical, enrichment-heavy timing is retained only for context: the old full
matrix took 2,455.3 seconds, and its 98-file noisy-corpus step took 335.5
seconds (roughly 0.29 files/s). Those figures included model-backed keyword,
entity, and hierarchy work and are not source-index throughput.

Required-suite queries averaged 4.1 seconds with a 3.6-second median. The
slowest limitation query took 27.9 seconds.

## NapierOne Real-File Ingestion

These 2026-07-28 runs used unchanged, SHA-256-verified files from NapierOne.
They measured parsing, immediate source indexing, SQLite storage, no-change
re-pointing, and hard-crash recovery. They did not measure retrieval relevance
or optional Qwen enrichment.

### Clean Runs

| Slice | Discovered | Indexed | Failures | Source size | Query-ready | Files/s | MiB/s | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CSV/TXT/JSON/JavaScript/HTML/XML | 606 | 606 | 0 | 54.9 MB | 55.43s | 10.93 | 0.945 | 253 MB |
| PDF/DOCX/PPTX | 303 | 293 | 10 | 490.8 MB | 86.20s | 3.40 | 5.385 | 615 MB |
| CSV/JSON/JavaScript/HTML/XML scale | 5,005 | 4,994 | 11 | 523.6 MB | 687.19s | 7.27 | 0.715 | 262 MB |

Storage and unchanged re-point measurements:

| Slice | Indexed source | SQLite size | SQLite/source | No-change re-point |
|---|---:|---:|---:|---:|
| Text/code | 54.9 MB | 94.7 MB | 1.72x | 0.182s |
| Rich documents | 486.7 MB | 26.6 MB | 0.055x | 0.693s |
| Scale | 515.6 MB | 1.43 GB | 2.78x | 3.128s |

The rich slice indexed all 100 DOCX files. Eight PDFs had no embedded text and
two PPTX files had no text shapes, producing a visible 3.3% failure rate rather
than empty searchable documents.

The scale slice indexed all 4,000 JSON, JavaScript, HTML, and XML files. Ten
CSV exports had blank/title rows before their usable header, and one CSV had a
16,383-field first row beyond SQLite's 2,000-column limit. Those files require
user cleanup or reshaping.

### Crash Recovery

| Slice | Forced exit after | Resume time | Final inventory/counts | Orphan raw records | Gate |
|---|---:|---:|---|---|---|
| Text/code | 10 durable files | 46.50s | exact clean-run match | none | pass |
| Rich documents | 10 durable files | 80.67s | exact clean-run match | none | pass |
| Scale | 100 durable files | 668.79s | exact clean-run match | none | pass |

Every slice exceeded the one-file-per-second target. `query_ready` permits a
partial collection only when every failure is explicit and no supported file
remains pending; it is not the same as a completely healthy collection.

### Extreme File Count

An ordinary unchanged `point()` over 522,931 tiny projected Quora files did
not finish within a 20-minute diagnostic window because it still walks and
hashes each source file. A benchmark-only strict persisted-index validation
finished in about 5.5 seconds. The latter is not public product behavior; this
remains an extreme-file-count startup limitation rather than a query-latency
measurement.

## Broad BEIR Component Ablation

Run date: 2026-07-30. Git commit:
`2893be4f35cacb67c8ca8627b20f08cf1dfd9817` from a clean worktree.

- 66,454 source documents: 3,633 NFCorpus, 57,638 FiQA, and 5,183 SciFact.
- All 1,271 judged test queries: 323 NFCorpus, 648 FiQA, and 300 SciFact.
- Source-only indexes; optional document enrichment disabled.
- Four paired variants with identical indexes, query order, candidate budgets,
  compiler, and exact Pyrrho runtime.
- INT8 `Alibaba-NLP/gte-reranker-modernbert-base` and managed Qwen.
- 2,000-sample paired percentile bootstrap intervals.
- Measurement-integrity gate passed. Total four-variant wall time was about
  6.3 hours.

Variants:

| Variant | Managed Qwen query terms | INT8 cross-encoder |
|---|---:|---:|
| `literal` | off | off |
| `expansion` | on | off |
| `reranker` | off | on |
| `full` | on | on |

### Final nDCG@10

| Dataset | Plain BM25 | `literal` | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|---:|
| NFCorpus | 0.3062 | 0.3179 | 0.3265 | 0.3392 | 0.3377 |
| FiQA | 0.2377 | 0.2490 | 0.2459 | 0.3170 | 0.3188 |
| SciFact | 0.6634 | 0.6457 | 0.6438 | 0.6556 | 0.6529 |
| Macro mean | n/a | 0.4042 | 0.4054 | 0.4372 | 0.4365 |

### Delivered nDCG@10

| Dataset | Plain BM25 reference | `literal` | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|---:|
| NFCorpus | 0.3062 | 0.3172 | 0.3257 | 0.3382 | 0.3367 |
| FiQA | 0.2377 | 0.2480 | 0.2447 | 0.3160 | 0.3179 |
| SciFact | 0.6634 | 0.6053 | 0.6045 | 0.6191 | 0.6170 |
| Macro mean | n/a | 0.3902 | 0.3916 | 0.4244 | 0.4239 |

The BM25 column is a whole-document relevance reference. Plain BM25 does not
have Fitz-Sage compilation or delivery stages.

### Recall@50 Before Reranking

| Dataset | Plain BM25 | `literal` | Qwen-enabled |
|---|---:|---:|---:|
| NFCorpus | 0.2101 | 0.2164 | 0.2231 |
| FiQA | 0.4459 | 0.4746 | 0.4766 |
| SciFact | 0.8704 | 0.8952 | 0.8909 |

The literal Fitz-Sage path exceeded the benchmark-local whole-document BM25
Recall@50 on all three datasets. This isolates measurable value from typed
indexing, deterministic planning, and recall fusion without Qwen or the
cross-encoder.

### Component Effects And Latency

| Variant | Macro final nDCG@10 | Macro delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|
| `literal` | 0.4042 | 0.3902 | 2.07s |
| `expansion` | 0.4054 | 0.3916 | 4.03s |
| `reranker` | 0.4372 | 0.4244 | 4.18s |
| `full` | 0.4365 | 0.4239 | 6.15s |

Reranker effect with Qwen disabled:

| Dataset | Final nDCG@10 delta | Paired 95% interval |
|---|---:|---:|
| NFCorpus | +0.0213 | [+0.0088, +0.0334] |
| FiQA | +0.0680 | [+0.0495, +0.0863] |
| SciFact | +0.0099 | [-0.0287, +0.0471] |

Reranking improved macro final nDCG@10 by 0.0330 for 2.10 seconds mean added
latency. NFCorpus and FiQA gains were clear; SciFact was inconclusive.

Qwen effect without reranking, measured at recall nDCG@10:

| Dataset | Delta | Paired 95% interval |
|---|---:|---:|
| NFCorpus | +0.0087 | [+0.0031, +0.0152] |
| FiQA | -0.0027 | [-0.0087, +0.0034] |
| SciFact | -0.0005 | [-0.0109, +0.0101] |

With reranking active, Qwen changed macro final nDCG@10 by -0.0008 and
delivered nDCG@10 by -0.0005 while adding 1.98 seconds. This broad BEIR run did
not directly test ordinary company-document synonym and paraphrase bridges, so
it does not justify removing semantic expansion.

Full-pipeline mean latency was 5.98 seconds on NFCorpus, 5.58 seconds on FiQA,
and 6.90 seconds on SciFact. Lexical recall averaged roughly 0.16-0.38 seconds;
Qwen, reranking, and the exact Pyrrho decision dominated query time.

Operational boundaries observed in the run:

- All NFCorpus and SciFact documents indexed.
- FiQA contained 38 empty upstream records, including one judged-relevant
  record; all 38 were reported as unsearchable.
- Thirteen SciFact queries had a judged-relevant final candidate but no judged
  relevant compiled evidence. Every query contained a literal structured
  identifier. Identifier and abbreviation equivalence remains user-owned.
- Compiled and delivered rankings were identical on all three datasets.

## Frozen BEIR Semantic Holdout

Run date: 2026-07-30. Run ID: `1785447318-d0a05a18`. Git commit:
`d855e4bec89b7f74959902e3859b5647b564230d` from a clean worktree.

- Frozen manifest: `beir-semantic-vocabulary-holdout-v1`.
- 120 ArguAna and 120 Quora queries, selected before retrieval scores existed.
- 8,674 ArguAna documents and 522,931 Quora documents.
- 960 Fitz-Sage executions across the same four paired variants.
- Expansion model: `onnx-community/Qwen3-0.6B-DQ-ONNX`.
- Reranker: INT8 `Alibaba-NLP/gte-reranker-modernbert-base`.
- No resumed queries; all operational and integrity gates passed.
- Runtime: 6,975 seconds, about 1 hour 56 minutes.

### Dataset Results

| Dataset | Variant | Recall nDCG@10 | Final nDCG@10 | Delivered nDCG@10 | Recall@50 | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| ArguAna | plain BM25 | 0.4652 | n/a | n/a | 0.9250 | n/a |
| ArguAna | `literal` | 0.4509 | 0.4413 | 0.4260 | 0.9000 | 6.21s |
| ArguAna | `expansion` | 0.4463 | 0.4439 | 0.4311 | 0.9000 | 8.97s |
| ArguAna | `reranker` | 0.4509 | 0.4562 | 0.4433 | 0.9000 | 11.16s |
| ArguAna | `full` | 0.4463 | 0.4579 | 0.4445 | 0.9000 | 13.00s |
| Quora | plain BM25 | 0.7244 | n/a | n/a | 0.9245 | n/a |
| Quora | `literal` | 0.8043 | 0.8049 | 0.8049 | 0.9712 | 2.79s |
| Quora | `expansion` | 0.7878 | 0.7878 | 0.7878 | 0.9597 | 4.92s |
| Quora | `reranker` | 0.8043 | 0.8566 | 0.8566 | 0.9712 | 3.24s |
| Quora | `full` | 0.7878 | 0.8593 | 0.8593 | 0.9597 | 5.50s |

Macro averages:

| Variant | Recall nDCG@10 | Final nDCG@10 | Delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|
| `literal` | 0.6276 | 0.6231 | 0.6155 | 4.50s |
| `expansion` | 0.6171 | 0.6159 | 0.6095 | 6.94s |
| `reranker` | 0.6276 | 0.6564 | 0.6500 | 7.20s |
| `full` | 0.6171 | 0.6586 | 0.6519 | 9.25s |

### Component Effects

Without reranking, Qwen changed the two-dataset macro results by:

| Metric | Delta |
|---|---:|
| Recall nDCG@10 | -0.0106 |
| Final nDCG@10 | -0.0072 |
| Delivered nDCG@10 | -0.0060 |
| Mean latency | +2.44s |

The Quora recall delta was -0.0165 with a paired 95% interval of
[-0.0331, -0.0036], and its final delta was -0.0170 with an interval of
[-0.0325, -0.0039]. With reranking active, Qwen changed macro final nDCG@10 by
+0.0022 and delivered nDCG@10 by +0.0020 while adding 2.06 seconds; those
per-dataset effects were inconclusive.

Qwen generated 951 ArguAna phrases and 890 Quora phrases. A new token that
also appeared in a judged-relevant document occurred in 6.5% and 6.3% of
phrases respectively; 36/120 ArguAna queries and 22/120 Quora queries had any
such bridge token. The current model did not earn its cost on these two BEIR
tasks. The package still keeps semantic expansion as a best-effort broad
semantic-to-lexical bridge because BM25 alone remains lexical.

Without Qwen, the INT8 reranker changed final nDCG@10 by +0.0149 on ArguAna
(95% interval [-0.0504, +0.0799], +4.95 seconds) and +0.0518 on Quora (95%
interval [+0.0075, +0.1003], +0.45 seconds). The two-dataset macro gain was
+0.0333 final and +0.0345 delivered for +2.70 seconds.

Literal Fitz-Sage final nDCG@10 exceeded whole-document BM25 on short
question-like Quora (0.8049 versus 0.7244) and trailed it on long
passage-as-query ArguAna (0.4413 versus 0.4652). Long argument matching is a
measured retrieval boundary.

## Query Timing And Reranker Hardening

### Initial Bottleneck Diagnostic

The 2026-07-29 diagnostic used seed `20260729` and 12 warm queries per dataset.

| Dataset | Mean | p50 | p95 | Rerank share | Qwen share | Pyrrho share | Recall share |
|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | 25.51s | 17.22s | 61.27s | 79.4% | 10.4% | 6.8% | 2.1% |
| FiQA | 16.97s | 16.06s | 23.86s | 65.0% | 15.6% | 13.3% | 3.3% |
| SciFact | 22.72s | 21.00s | 40.23s | 70.9% | 14.8% | 10.0% | 2.0% |

Across all 36 queries, reranking consumed 72.7% of total time: 43.4% in the
initial pass and 29.3% in evidence-closure passes. The 20 closure queries
averaged 26.99 seconds versus 15.15 seconds without closure. Qwen averaged
2.89 seconds (13.3%); lexical recall averaged 0.52 seconds (2.4%).

### Accepted Hardening Result

The same 12-query SciFact sample was repeated after profile-aware candidate
budgets, two INT8 batch-one workers, exact deduplication, and a bounded score
cache. The model input limit was unchanged.

| SciFact warm metric | Before | After | Change |
|---|---:|---:|---:|
| Mean query latency | 22.72s | 7.82s | -65.6% |
| p50 query latency | 21.00s | 6.56s | -68.8% |
| p95 query latency | 40.23s | 18.66s | -53.6% |
| Mean rerank time | 16.11s | 2.95s | -81.7% |

A separate matched 60-query SciFact comparison used 18 broad, 38 moderate,
and 4 narrow profiles. The configured cross-encoder cap averaged 36.27
candidates while the full recall pool averaged 56.50.

| Matched 60-query metric | Before | After |
|---|---:|---:|
| Mean query latency | 20.84s | 7.43s |
| p50 query latency | 16.34s | 6.77s |
| p95 query latency | 38.65s | 12.56s |
| Reranked nDCG@10 | 0.7499 | 0.7577 |
| Reranked Recall@10 | 0.8600 | 0.8517 |
| Delivered nDCG@10 | 0.6709 | 0.6617 |
| Delivered Recall@10 | 0.7750 | 0.7750 |
| Queries with delivered relevant evidence | 47/60 | 47/60 |

The accepted tradeoff was -0.83 percentage points reranked Recall@10 and -0.92
points delivered nDCG@10, with unchanged delivered hit coverage. The required
boundary suite still passed 11/11 retrieval and delivery contracts afterward;
all 20 files indexed and enriched, and the production gate passed.

### INT8 Versus FP32 Decision Probe

This isolated paired probe used the same 60 SciFact queries, seed `20260729`,
batch size 1, and a mean 56.67 candidates. These scores are comparable only
inside this probe.

| Precision | Mean rerank | Median | p90 | Max | nDCG@10 | Recall@10 | Precision@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| INT8 | 4.204s | 4.010s | 5.799s | 6.231s | 0.6810 | 0.8167 | 0.5833 |
| FP32 | 7.598s | 7.390s | 10.081s | 12.314s | 0.7465 | 0.8333 | 0.6667 |

FP32 was 1.81x slower by mean rerank time and reached about 1.60 GB peak RSS
in the probe. It improved this probe's nDCG@10 by 0.0655 and Recall@10 by
0.0167. INT8 batch-one remains the accepted package choice: the measured
latency and memory cost were prioritized, with the quality tradeoff recorded
rather than hidden.

## What These Results Do Not Establish

- They do not prove that an arbitrary company folder works without data
  preparation.
- NapierOne establishes parser, storage, throughput, and recovery behavior,
  not whether retrieved evidence answers domain questions.
- BEIR establishes relevance behavior on its measured tasks, not production
  company-document quality or universal Qwen usefulness.
- The limitation suite is intentionally non-green at the full-contract level.
- Current Pyrrho results are not an independent governance-quality benchmark.
- Background Qwen document enrichment throughput, very large individual
  documents, OCR, and a public persisted-collection reuse path remain to be
  measured or implemented separately.

## Canonical Sources

- [Production readiness](docs/PRODUCTION_READINESS.md)
- [Limitations and benchmark interpretation](LIMITATIONS.md)
- [Benchmark runner and reranker validation](benchmarks/README.md)
- [BEIR component ablation](docs/evaluation/beir-component-ablation-2026-07-30.md)
- [Frozen BEIR semantic holdout](docs/evaluation/beir-semantic-holdout-2026-07-30.md)

Detailed machine-specific JSON, Markdown, stdout, and stderr artifacts are
under the ignored `benchmarks/results/` tree. The committed reports above are
the canonical durable summaries. When the remaining benchmarks are complete,
this temporary file should be reconciled against their release-candidate run
identities before selected claims move into the README.
