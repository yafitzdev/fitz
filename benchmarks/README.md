<!-- benchmarks/README.md -->
# Retrieval Benchmark

This benchmark is evidence-first. It does not score answer prose. It runs the
real folder-to-evidence path and validates returned evidence against
deterministic expectations.

Compact reports are the default. Each record includes:

- selected evidence identities, without source bodies
- query terms and query-shape signals
- compiled ranked evidence identities
- fixed Pyrrho-delivered evidence identities
- candidate counts at recall, reranking, and final-selection boundaries
- exact Pyrrho input and output
- stage attribution for missing evidence
- deterministic validation metrics and failures
- aggregate pass-rate summaries by domain and tag
- a per-file ingestion inventory, including unsupported and failed files

Use `--report-detail full` only when source-bearing debug artifacts are needed.
Full reports include the content-bearing `RetrievalRun` and can be large.

## Run

```bash
python -m benchmarks.fitz_bench.runner
```

To run against an unpacked local Pyrrho package:

```bash
python -m benchmarks.fitz_bench.runner --governance "pyrrho/path/to/model"
```

Defaults:

- corpus: `benchmarks/corpora/core`
- cases: `benchmarks/cases/core.yaml`
- JSON report: `benchmarks/results/latest.json`
- Markdown summary: `benchmarks/results/latest.md`
- workspace: `.bench_workspace/<collection>`
- index mode: `complete` (use `source` to measure retrieval without enrichment)
- report detail: `compact`
- gate: all assertions, including governance

For a quick smoke run:

```bash
python -m benchmarks.fitz_bench.runner --limit 2 --index-mode progressive
```

Measure cold query-ready ingestion separately from optional Qwen enrichment:

```bash
python -m benchmarks.fitz_bench.ingestion_benchmark \
  --source benchmarks/corpora/core \
  --iterations 3 \
  --target-files-per-second 1
```

The report contains `query_ready_seconds`, `files_per_second`, indexing
failures, and no-change re-point time. It always calls `point(...,
start_worker=False)`, so model download, entity extraction, and hierarchy
summaries cannot distort the source-index throughput metric.

## External Corpus Ingestion

The production ingestion benchmark uses
[NapierOne](https://registry.opendata.aws/napierone/), not a generated fixture
corpus. NapierOne contains more than 500,000 real files across 44 file types
and is available from a public AWS bucket without an AWS account.

Fitz-Sage does not redistribute or rewrite these files. The benchmark helper:

- downloads only the requested official archives;
- verifies every archive against NapierOne's published SHA-256 report;
- rejects unsafe ZIP paths, symlinks, and configured download/extraction
  budget overruns;
- extracts archives byte-for-byte into `.benchmark-data/napierone`;
- creates hard-linked selection views where supported, with a byte-copy
  fallback, so a run sees exactly its requested types without modifying source
  bytes;
- keeps all corpus data, workspaces, and generated reports out of Git.

Run the default tiny slice of supported document types:

```bash
python -m benchmarks.fitz_bench.external_ingestion_benchmark \
  --profile tiny \
  --workspace .bench_workspace/napierone
```

Download and verify without indexing:

```bash
python -m benchmarks.fitz_bench.external_ingestion_benchmark \
  --profile small \
  --type PDF \
  --type DOCX \
  --download-only
```

Re-run an already cached selection without network access:

```bash
python -m benchmarks.fitz_bench.external_ingestion_benchmark \
  --profile tiny \
  --type PDF \
  --type DOCX \
  --type PPTX \
  --offline
```

NapierOne publishes `tiny`, `small`, and `total` archives containing 100,
1,000, and 5,000 examples per selected type. Start with explicit `--type`
arguments and byte budgets before using a larger profile. The full upstream
dataset approaches 2 TB.

Each run records throughput, source-size distribution, per-extension outcomes,
SQLite retrieval-unit counts, storage amplification, peak process RSS,
unchanged re-point idempotence, and parser failure details. Unless
`--skip-recovery` is set, a child process is terminated with `os._exit()` after
a configurable number of durable file writes. The parent resumes the same
collection and requires its manifest and SQLite counts to match a clean run
exactly.

The files are external and should be treated as untrusted input. The benchmark
parses supported documents; it does not execute source files. NapierOne is
provided under the Edinburgh Napier University License Agreement and requires
source attribution. See the upstream
[license and attribution terms](https://github.com/simonrdavies/NapierOne#license-and-attribution).
NapierOne Mixed File Dataset was accessed on 2026-07-28 from
https://registry.opendata.aws/napierone/.

Generated reports under `benchmarks/results/` and benchmark workspaces under
`.bench_workspace/` are ignored by git.

## External Labeled Retrieval

The BEIR benchmark measures document retrieval against upstream relevance
judgments instead of Fitz-Sage-authored expectations. The selected datasets
cover three materially different domains:

- NFCorpus: biomedical nutrition questions, 3,633 documents and 323 test
  queries
- FiQA: financial question answering, 57,638 documents and 648 test queries
- SciFact: scientific claim retrieval, 5,183 documents and 300 test queries

The helper downloads the official BEIR archives, verifies their published MD5
values, safely extracts them, and validates the expected corpus and judged
query counts. It does not redistribute the datasets.

BEIR stores each document as `title`, `text`, and metadata. The adapter creates
one UTF-8 `.txt` source per document containing the exact title, a blank line,
and the exact text. It does not summarize, normalize, expand, or include BEIR
metadata. Unsafe document IDs never become paths: a SHA-256 filename and an
external reversible mapping preserve identity without adding benchmark labels
to the indexed corpus.

Run all three datasets through a transparent Okapi BM25 baseline and
Fitz-Sage's canonical traced retrieval:

```bash
python -m benchmarks.fitz_bench.beir_benchmark
```

Measure the contribution and cost of query expansion and reranking without
changing production configuration:

```bash
python -m benchmarks.fitz_bench.beir_ablation \
  --offline
```

The ablation runs four query-side configurations against the same reusable
indexes:

- `literal`: deterministic query planning and typed lexical recall, with
  managed Qwen keywords disabled and cross-encoder scoring replaced by stable
  top-k selection
- `expansion`: `literal` plus managed Qwen semantic query keywords
- `reranker`: `literal` plus the canonical INT8 cross-encoder
- `full`: the canonical pipeline with both components

The stable selector preserves the reranker's top-k output budget, so read,
evidence-closure, compilation, and Pyrrho inputs remain structurally
comparable. Each variant execution runs in a fresh process; resumed checkpoints
can combine completed query records from more than one process lifetime. The
aggregate report aligns records by query ID and gives paired 95% bootstrap
intervals for recall, final-candidate quality, delivered-evidence quality, and
latency. Variant checkpoints are separate and resumable. Use
`--no-resume-queries` to replace them after intentionally changing retrieval
behavior.

Download, verify, and project the corpora without indexing:

```bash
python -m benchmarks.fitz_bench.beir_benchmark --download-only
```

Run a quick SciFact smoke from the verified local cache:

```bash
python -m benchmarks.fitz_bench.beir_benchmark \
  --dataset scifact \
  --offline \
  --query-limit 3
```

The default is source-only indexing. Use `--index-mode complete` to measure the
same queries after optional entity and hierarchy enrichment.
`--reuse-workspace` keeps dataset indexes across interrupted runs. Add
`--resume-queries` to reuse matching per-query checkpoints; checkpoint
signatures include the archive, query set, cutoffs, mode, governance selection,
and retrieval source digest, so stale results are rejected.
`--query-limit` is a smoke-test control and must not be used for release
claims.

Each query reports rankings and TREC-style precision, recall, MRR, MAP, and
graded nDCG at 1, 3, 5, 10, 20, and 50 for:

- benchmark-local plain BM25
- Fitz-Sage recall
- raw reranker output
- final candidate selection
- compiled ranked evidence
- delivered evidence

Repeated chunks from one source count as one document. A delivered miss is
attributed to the earliest irreversible boundary: recall, final selection,
evidence compilation, or delivery. The exact generated query terms and Pyrrho
decision are retained per query. Pyrrho does not alter relevance scores and is
not a retrieval gate.

The default gate is operational only: archive, indexing, identity-mapping, and
query-execution failures fail the run, while low scores remain measured
limitations. A score gate can be introduced explicitly with
`--max-recall-regression`, but it should be based on an accepted measured
baseline rather than selected before the first run.

BEIR is a benchmark wrapper, not one license. Users remain responsible for the
license of each underlying dataset. See the
[BEIR repository](https://github.com/beir-cellar/beir) and
[BEIR paper](https://arxiv.org/abs/2104.08663).

### Frozen Semantic Vocabulary Holdout

The broad three-dataset suite does not isolate the intended job of managed
query expansion. The committed
`benchmarks/fixtures/beir_semantic_holdout_v1.json` adds two established BEIR
tasks:

- ArguAna: retrieve the best counterargument from 8,674 arguments
- Quora: retrieve duplicate questions from 522,931 questions

The fixture contains 120 test queries from each dataset. It was frozen before
running Fitz-Sage retrieval. Selection uses no Fitz-Sage, BM25, Qwen, reranker,
or Pyrrho output:

1. For every judged query, compute the maximum case-folded token-set Jaccard
   overlap against its available judged-relevant documents.
2. Sort all eligible queries by overlap and split them into low, medium, and
   high lexical-overlap tertiles.
3. Select 40 queries per tertile with a deterministic SHA-256 ordering using
   seed `20260730`.

The manifest records the official archive checksum, extracted source hashes,
selection ranges, and exact query IDs. Quora and ArguAna evaluation excludes a
corpus result whose ID equals the query ID, matching their BEIR evaluation
contract.

The official ArguAna archive has five positive qrel targets that are absent
from its corpus. The manifest records all five query/document pairs and makes
those queries ineligible. It does not infer replacement IDs. Quora has no
missing positive targets.

Regenerate and compare the fixture without retrieval:

```bash
python -m benchmarks.fitz_bench.beir_holdout \
  --offline
```

Run the paired four-variant measurement:

```bash
python -m benchmarks.fitz_bench.beir_ablation \
  --offline \
  --query-manifest benchmarks/fixtures/beir_semantic_holdout_v1.json \
  --no-resume-queries
```

Each ablation child reuses a persisted index only after validating its source
path, expected searchable-document count, ingestion failures, source-ID
mapping, and every persisted content hash against the deterministic adapter
mapping. This avoids repeating a full no-change `point()` traversal for every
variant without accepting mismatched or partial persisted state.

The aggregate report gives the same paired component effects as the broad
suite and additionally reports Qwen effects separately for the frozen low,
medium, and high lexical-overlap strata.

The completed run evaluated all 240 queries under all four variants. All child
operational gates and the paired-integrity gate passed, with no resumed
queries.

| Dataset | `literal` final nDCG@10 | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|
| ArguAna | 0.4413 | 0.4439 | 0.4562 | 0.4579 |
| Quora | 0.8049 | 0.7878 | 0.8566 | 0.8593 |

Without reranking, Qwen reduced macro recall nDCG@10 by 0.0106 and final
nDCG@10 by 0.0072 while adding 2.44 seconds. Its Quora recall and final
regressions were conclusive, and no consistent benefit appeared in the frozen
low-overlap strata. With reranking active, Qwen changed macro final nDCG@10 by
only +0.0022 while adding 2.06 seconds.

Without Qwen, the reranker improved Quora final nDCG@10 by 0.0518 for 0.45
seconds. Its ArguAna gain was inconclusive and cost 4.95 seconds, showing why
reranker quality and latency must be reported by query shape.

The current managed expansion path did not earn its cost on these two BEIR
tasks. This is not a product-wide reason to disable Qwen: BM25 remains lexical,
and the holdout is a proxy rather than an application-shaped company-document
test. The holdout remains frozen and must not be used to tune term filters or
fusion. Full methodology, paired intervals, per-query diagnostics, and
operational findings are recorded in
[`docs/evaluation/beir-semantic-holdout-2026-07-30.md`](../docs/evaluation/beir-semantic-holdout-2026-07-30.md).

### Measured BEIR Baseline

The 2026-07-30 source-only run evaluated all 1,271 judged test queries from a
clean `2893be4f` worktree. Optional per-document enrichment was disabled. The
canonical `full` variant used managed Qwen query terms, the INT8 ONNX
cross-encoder, evidence compilation, and exact Pyrrho integration.

Mean nDCG@10 by observable stage:

| Dataset | Plain BM25 | Fitz recall | Final candidates | Delivered evidence |
|---|---:|---:|---:|---:|
| NFCorpus | 0.3062 | 0.3264 | 0.3377 | 0.3367 |
| FiQA | 0.2377 | 0.2445 | 0.3188 | 0.3179 |
| SciFact | 0.6634 | 0.6403 | 0.6529 | 0.6170 |

Recall@50 before the fixed top-10 reranking/evidence window:

| Dataset | Plain BM25 | Fitz literal recall | Qwen-enabled recall |
|---|---:|---:|---:|
| NFCorpus | 0.2101 | 0.2164 | 0.2231 |
| FiQA | 0.4459 | 0.4746 | 0.4766 |
| SciFact | 0.8704 | 0.8952 | 0.8909 |

The literal Fitz-Sage path exceeded whole-document BM25 Recall@50 on all three
datasets without Qwen terms. This isolates the benefit of typed indexing,
deterministic query planning, and retrieval fusion from managed semantic
expansion.

The paired four-variant ablation measured:

| Variant | Macro final nDCG@10 | Macro delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|
| `literal` | 0.4042 | 0.3902 | 2.07s |
| `expansion` | 0.4054 | 0.3916 | 4.03s |
| `reranker` | 0.4372 | 0.4244 | 4.18s |
| `full` | 0.4365 | 0.4239 | 6.15s |

With Qwen disabled, reranking improved macro final nDCG@10 by 0.0330 for a
2.10-second mean cost. Its paired gain was conclusive on NFCorpus and FiQA and
inconclusive on SciFact. Qwen improved NFCorpus recall nDCG@10 by 0.0087, with
a paired 95% interval of [+0.0031, +0.0152], but had no conclusive recall gain
on FiQA or SciFact. With reranking active, it changed macro final nDCG@10 by
-0.0008 while adding 1.98 seconds.

This broad suite alone was not enough to judge semantic expansion because it
did not target vocabulary mismatch. The separate frozen ArguAna/Quora holdout
described above now supplies a stronger BEIR-specific follow-up. Qwen did not
earn its cost on those tasks, but the result does not settle its value for
ordinary company-document retrieval.

All 3,633 NFCorpus and 5,183 SciFact documents indexed. FiQA indexed
57,600/57,638 documents; the 38 failures had empty upstream `title` and `text`
fields, and one empty record was nevertheless judged relevant. Empty documents
remained visible failures and were not removed by the adapter.

Thirteen SciFact queries had a judged-relevant final candidate but no judged
relevant compiled evidence. Every query contained a literal structured
identifier such as `anti-interleukin-2`, `SHP-2`, `CK-666`, or `FOXO3`.
Fitz-Sage intentionally does not equate separator, abbreviation, or naming
variants. That equivalence belongs to user cleanup or preprocessing. A public
vocabulary hook is deferred and is not part of the current API.

Canonical query latency averaged 5.98 seconds on NFCorpus, 5.58 seconds on
FiQA, and 6.90 seconds on SciFact. The local plain-BM25 comparison excludes
managed models, compilation, and Pyrrho and is not a latency-equivalent product
path. Full methodology, per-dataset confidence intervals, and timing details
are recorded in
[`docs/evaluation/beir-component-ablation-2026-07-30.md`](../docs/evaluation/beir-component-ablation-2026-07-30.md).

Profile the existing reusable indexes without repeating ingestion or the full
relevance run:

```bash
python -m benchmarks.fitz_bench.beir_timing --offline
```

The timing profiler runs one explicit cold query per process and then a
deterministic warm sample. It preserves raw `EvidencePack` timings and also
groups non-overlapping query preparation, semantic expansion, recall,
reranking, reading, context expansion, and Pyrrho costs. Aggregate retrieval
timers are excluded from the grouped totals to avoid double-counting.

The 2026-07-29 diagnostic run used seed `20260729` and 12 warm queries per
dataset on the same six-core benchmark machine:

| Dataset | Mean | p50 | p95 | Rerank | Qwen terms | Pyrrho | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| NFCorpus | 25.51s | 17.22s | 61.27s | 79.4% | 10.4% | 6.8% | 2.1% |
| FiQA | 16.97s | 16.06s | 23.86s | 65.0% | 15.6% | 13.3% | 3.3% |
| SciFact | 22.72s | 21.00s | 40.23s | 70.9% | 14.8% | 10.0% | 2.0% |

Across the 36 warm queries, reranking consumed 72.7% of total time: 43.4%
for the initial 50-candidate pass and 29.3% for repeated evidence-closure
passes. Twenty queries requested closure; they averaged 26.99 seconds versus
15.15 seconds without closure. Qwen query expansion averaged 2.89 seconds
(13.3%), while lexical recall averaged 0.52 seconds (2.4%). This identifies
cross-encoder reranking and closure fan-out as the latency bottleneck; corpus
size and BM25 are not the primary cause.

### Reranker Hardening Validation

The 2026-07-29 reranker hardening run repeated the same seeded 12-query
SciFact timing sample after introducing profile-aware candidate budgets,
two INT8 batch-one workers, exact deduplication, and a bounded score cache.
The 512-token model input limit was unchanged.

| SciFact warm metric | Before | After | Change |
|---|---:|---:|---:|
| Mean query latency | 22.72s | 7.82s | -65.6% |
| p50 query latency | 21.00s | 6.56s | -68.8% |
| p95 query latency | 40.23s | 18.66s | -53.6% |
| Mean rerank time | 16.11s | 2.95s | -81.7% |

A separate 60-query SciFact run matched every query ID against the
pre-change full report. It used 18 broad, 38 moderate, and 4 narrow profiles,
for a mean configured cross-encoder cap of 36.27 candidates while preserving
a mean full recall pool of 56.50.

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

This is a latency optimization with a measured quality tradeoff, not a quality
claim. Reranked ordering improved slightly on aggregate, reranked recall
declined by 0.83 percentage points, and delivered ordering declined by 0.92
points while delivered hit coverage remained unchanged.

The required 11-case `hardened_boundaries` suite was then rerun against the
limits corpus. Retrieval and delivered-evidence assertions both passed 11/11,
all 20 files indexed and enriched, and the production gate passed. Full
governed contracts passed 5/11; those accepted Pyrrho outcomes remain separate
from the retrieval gate.

These results are a measured boundary, not an SLA and not an official BEIR
leaderboard submission. The benchmark-local analyzer and one-file adapter are
fully described above, and no retrieval behavior was changed in response to
the scores during the run.

## Case Shape

```yaml
- id: code_symbol_lookup
  domain: code
  query: "Where is expired session refresh implemented?"
  expected:
    mode: sufficient
    required_evidence:
      - file: code/auth_service.py
        kind: symbol
        location_contains: "refresh_expired_session"
        contains: ["refresh_expired_session", "grace"]
    forbidden_evidence:
      - file: docs/stale.md
        contains: ["old behavior"]
  tags: [code, symbol_lookup]
```

Cases use the v2 evidence-verdict names: `sufficient`, `disputed`, and
`insufficient`.

The validator checks evidence items, not generated text. A required evidence
entry passes when any returned evidence item matches the file/kind/location and
contains every listed text fragment.

## Suites

The starter corpus covers:

- unstructured prose
- structured CSV/table evidence
- code symbols
- mixed table/prose cases
- conflicts and insufficient evidence
- temporal freshness and stale evidence
- explicit in-corpus acronym bridges
- filtered table lookups and comparisons
- stale documentation versus implementation conflicts
- cross-domain evidence closure

Available suites:

- Core: `benchmarks/corpora/core` with `benchmarks/cases/core.yaml`
- Holdout: `benchmarks/corpora/holdout` with `benchmarks/cases/holdout.yaml`
- Holdout2: `benchmarks/corpora/holdout2` with `benchmarks/cases/holdout2.yaml`

Run Holdout2 explicitly:

```bash
python -m benchmarks.fitz_bench.runner --corpus benchmarks/corpora/holdout2 --cases benchmarks/cases/holdout2.yaml --output benchmarks/results/holdout2_latest.json --markdown benchmarks/results/holdout2_latest.md
```

Grow this by adding more files under `benchmarks/corpora/` and YAML cases under
`benchmarks/cases/`.

For focused diagnosis, repeat `--case-id` without editing the suite:

```bash
python -m benchmarks.fitz_bench.runner \
  --corpus benchmarks/corpora/limits \
  --cases benchmarks/cases/limits.yaml \
  --case-id structured_large_rec0619_owner \
  --case-id conflict_run55b_final_audit
```

The runner prints one progress line per completed case. The full limitations
suite exercises managed Qwen, reranking, evidence closure, and Pyrrho for every
query, so it is a release-gate run rather than a fast smoke test.

## Production Matrix

The production matrix composes standard, holdout, corpus-growth, format,
reload-stability, optional-format, and intentional-limitations suites:

```bash
python -m benchmarks.fitz_bench.production_runner \
  --output production-report.json
```

The required matrix currently contains:

- core, holdout, and second holdout corpora
- an 80-document near-neighbor corpus-growth run
- reload stability over the grown corpus
- 60 explicit temporal, comparison, aggregation, and narrow query-shape cases
- PDF, DOCX, PPTX, SQL, Python, Go, Java, and TypeScript evidence
- an 11-case required hardening gate for difficult retrieval boundaries
- measured, non-gating XLSX and known-limitations suites

Required retrieval suites gate at 85%. The query-shape suite gates its own
signals at 85%. Corpus growth fails the production gate if a shared case
regresses even when the aggregate rate remains above threshold. Pyrrho outcomes
remain visible but are measured separately because model evaluation belongs to
Pyrrho's own release lifecycle. Supported-file ingestion failures always fail
a required suite.

Run one suite while developing:

```bash
python -m benchmarks.fitz_bench.production_runner \
  --suite-id hardened_boundaries
```

## Metric Boundaries

- `retrieval_pass_rate`: required evidence is present and forbidden evidence is
  absent in the compiled ranking.
- `delivery_pass_rate`: the same evidence assertions over the fixed delivered
  `EvidencePack`.
- `query_shape_pass_rate`: explicit query-plan signals match their human-labeled
  temporal, comparison, aggregation, or narrow expectation.
- `capability_pass_rate`: all evaluated retrieval and
  query-shape assertions pass.
- `pass_rate`: governed delivery, query shape, and expected Pyrrho mode all
  pass.
- `retrieval_stability_rate`: compiled ranked identities survive a fresh
  engine load.
- `delivery_stability_rate`: delivered evidence identities survive a fresh
  engine load.
- `governance_stability_rate`: the Pyrrho mode survives a fresh engine load.

Every rate includes an `*_evaluated` denominator. Cases without an assertion
for that metric do not receive automatic credit. Do not use the full pass rate
to describe retrieval quality. A mode-only Pyrrho error is reported as a
Pyrrho failure; a correct item excluded by the fixed budget is a delivery
failure, not a recall failure.
