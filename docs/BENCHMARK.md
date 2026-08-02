# Fitz-Sage Benchmarks

This is the canonical benchmark report for the current Fitz-Sage retrieval
architecture. Last consolidated: 2026-08-02.

It records accepted measurements, methodology, component ablations, and the
diagnostics that explain current design decisions. Smoke runs, interrupted
runs, and superseded intermediate reports are excluded. Measurements were made
at the revisions identified in each section; they are a capability record, not
one release-candidate scorecard.

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
| Enterprise retrieval holdout | 511,961 docs, 328 queries | full delivered nDCG@10 0.5780; reranker-only 0.5876 |
| SciFact query latency | matched 60-query sample | 7.43s mean; 6.77s p50; 12.56s p95 |
| Enterprise warm query probes | 511,961-file index | 13.092s and 19.889s |

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
  compiler, and exact pinned Pyrrho model.
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

## Frozen EnterpriseRAG-Bench Holdout

Run completion: 2026-08-01. Aggregate holdout run ID:
`1785549941-8fea6366`. Git commit:
`d6ac9799ef4a0b55058913a40d899e15ce2e5d3e` from clean child-run
worktrees.

- Official EnterpriseRAG-Bench `v1.0.0` corpus.
- 511,962 source files across Confluence, Fireflies, GitHub, Gmail, Google
  Drive, HubSpot, Jira, Linear, and Slack.
- 511,961 files evaluated after explicitly excluding one Defender-quarantined
  synthetic safety file that was irrelevant to every question.
- 470 scored questions split before retrieval into 142 development and 328
  untouched holdout questions.
- Frozen split digest:
  `74c2d04b613430f89052fa8a8f5fb9853e6f310289d2c9faa9381d89aaeabba5`.
- Original UTF-8 document bytes and source hierarchy; no source rewriting,
  normalization, summaries, generated metadata, or benchmark labels.
- Source-only index, so optional document enrichment remained pending.
- 1,312 holdout Fitz-Sage executions across the same four paired variants.
- All source, operational, and paired-integrity gates passed.
- Holdout wall time was 51,256.8 seconds, about 14 hours 14 minutes.

Holdout quality:

| Variant | Broad Recall@50 | Final Recall | Final nDCG@10 | Delivered Recall | Delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| Plain whole-document BM25 | 0.8092 | n/a | n/a | n/a | n/a | 2.49s |
| `literal` | 0.7824 | 0.6569 | 0.5276 | 0.6450 | 0.5279 | 29.14s |
| `expansion` | 0.7801 | 0.6611 | 0.5311 | 0.6491 | 0.5324 | 32.82s |
| `reranker` | 0.7824 | 0.7006 | 0.5768 | 0.6883 | 0.5876 | 39.49s |
| `full` | 0.7801 | 0.6936 | 0.5629 | 0.6816 | 0.5780 | 44.20s |

The plain-BM25 latency excludes query planning, models, evidence processing,
and Pyrrho. Its broad Recall@50 exceeded literal Fitz-Sage by 0.0269, which is
a measured source-only recall boundary on this corpus.

Paired holdout effects:

| Change | Final Recall delta | Final nDCG@10 delta | Delivered nDCG@10 delta | Added latency |
|---|---:|---:|---:|---:|
| Qwen, no reranker | +0.0041 `[-0.0165, +0.0244]` | +0.0034 `[-0.0047, +0.0120]` | +0.0045 `[-0.0042, +0.0136]` | 3.68s |
| Reranker, no Qwen | +0.0437 `[+0.0057, +0.0825]` | +0.0492 `[+0.0125, +0.0875]` | +0.0597 `[+0.0238, +0.0988]` | 10.35s |
| Full versus literal | +0.0366 `[-0.0017, +0.0785]` | +0.0352 `[-0.0029, +0.0721]` | +0.0501 `[+0.0147, +0.0851]` | 15.06s |

The INT8 reranker gain generalized. Full versus literal had a positive
delivered-ranking gain, while its final recall and nDCG intervals crossed
zero. Qwen's isolated aggregate quality effect was inconclusive. Qwen remains
the intentional broad semantic-to-lexical recall leg; this score does not
justify narrowing candidate competition or removing expansion.

The clearest architecture weakness was multi-document ranking. Reranker-only
final nDCG@10 improved from 0.5192 to 0.5892 on 265 single-document questions,
but declined from 0.5633 to 0.5247 on 63 multi-document questions.
Project-related questions showed a conclusive `-0.0796` full-versus-literal
final nDCG@10 delta. This points to future set-aware coverage after pointwise
reranking, not source cleanup, alias heuristics, or governance logic.

Latency was also a clear boundary. In the full variant, managed Qwen averaged
1.95 seconds and reranking 7.62 seconds, while initial and repeated
evidence-closure recall averaged 29.27 seconds. Recall orchestration is the next
latency target. One reranker-only query had a retained 344.61-second transient
outlier, of which 330.45 seconds was repeated recall rather than cross-encoder
work.

### Evidence-Closure Latency Investigation

The persisted EnterpriseRAG-Bench source-only index contained 666,785 section
rows and no table or symbol rows. Despite that physical boundary, the full
holdout run executed closure retrieval for 241/328 queries and made 834 closure
passes. Those passes consumed 6,934.28 seconds in total, or 21.14 seconds per
holdout query. Pyrrho required an unavailable table or symbol modality for
238 queries; the contracts imply at least 363 passes that could not possibly
return the requested evidence from this index.

The executor now checks which physical modalities exist before running a
closure request, skips only requests whose requested index is empty, and
records each skip in the retrieval trace. Executed closure searches use only
the request-local terms and requested retrieval strategy instead of replaying
the original query's semantic, comparison, and temporal recall legs. Pyrrho's
plan and verdict are not changed.

Section FTS was also changed to rank lightweight row IDs before materializing
the winning section bodies. On the 7.75 GB Enterprise index, a matched 64-row
search fell from 5.9821 seconds to 3.0509 seconds with identical result IDs.
Two matched warm query probes changed as follows:

| Enterprise query probe | Before | After | Closure behavior after |
|---|---:|---:|---|
| Unavailable symbol/table obligations (`qst_0169`) | 71.229s | 13.092s | 6/6 requests traced and skipped |
| Valid section obligation (`qst_0164`) | 35.136s | 19.889s | 1/1 request executed and added evidence |

These are diagnostic probes, not a rerun of holdout quality. The post-change
`hardened_boundaries` gate preserved retrieval and delivered evidence at 11/11;
the production gate passed. Full governed contracts passed 6/11, with all five
failures attributed to the accepted Pyrrho model.

Qwen produced malformed optional expansion output for 11/328 holdout queries.
Every query completed through the literal plan, with the failure logged and
traced. The availability fallback was implemented on development before the
holdout ran and does not alter Pyrrho or invent replacement terms.

Full methodology, category/source breakdowns, development results, confidence
intervals, and boundaries are in
[`evaluation/enterprise-rag-bench-2026-08-01.md`](evaluation/enterprise-rag-bench-2026-08-01.md).

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
  documents, OCR, and a fast validated no-change re-point path remain to be
  measured or implemented separately.

## Canonical Sources

- [Production readiness](PRODUCTION_READINESS.md)
- [Limitations and benchmark interpretation](LIMITATIONS.md)
- [Benchmark runner and reranker validation](../benchmarks/README.md)
- [BEIR component ablation](evaluation/beir-component-ablation-2026-07-30.md)
- [Frozen BEIR semantic holdout](evaluation/beir-semantic-holdout-2026-07-30.md)
- [Frozen EnterpriseRAG-Bench holdout](evaluation/enterprise-rag-bench-2026-08-01.md)

Detailed machine-specific JSON, Markdown, stdout, and stderr artifacts are
under the ignored `../benchmarks/results/` tree. The committed reports above are
the durable run summaries. Update this report when an accepted rerun changes a
current measurement or a documented product boundary.
