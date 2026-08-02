# Production Readiness

Fitz-Sage's production contract is:

> Point Fitz-Sage at a folder of reasonably clean, supported company documents
> and retrieve grounded source evidence for ordinary business questions without
> project-specific retrieval code.

This is narrower than "arbitrary files work" and stronger than a collection of
isolated unit tests.

## Responsibilities

Fitz-Sage owns:

- recursive discovery and explicit supported/unsupported/failure states;
- parsing into searchable sections, symbols, and native tables;
- literal and managed semantic query recall;
- temporal, comparison, aggregation, and other general query-shape handling;
- BM25 recall, reranking, source reading, evidence closure, and compilation;
- provenance and inspectable retrieval-run records;
- exact transport of Pyrrho planning and final governance output.

The user owns:

- OCR/parser selection for content the default parser cannot read;
- raw-log compression, rewriting, and domain cleanup;
- private acronym, synonym, and identifier mappings;
- deciding whether differently written identifiers are equivalent;
- removing secrets and documents that must not be indexed;
- application-specific synthesis, UI, and workflow policy.

See [Limitations](LIMITATIONS.md) for the detailed boundary.

## Source-Index Contract

`point()` returns only after every supported file is searchable or explicitly
failed. Optional entity, hierarchy, and demand-summary work has separate status
and may continue afterward.

- `query_ready`: no supported file remains pending.
- `complete`: every supported file indexed successfully.
- `healthy`: no supported-file failure was recorded.
- `enrichment.complete`: optional derived work has settled.

A partial collection can be query-ready without being complete or healthy. That
allows useful retrieval without representing failures as success.

## Release Gates

The internal production runner starts from source folders and measures stages
separately. It does not inject prebuilt chunks or retrieval results.

Current gates include:

| Gate | Requirement |
|---|---|
| Required compiled retrieval | at least 85% |
| Required fixed evidence delivery | at least 85% |
| Query-shape recognition | at least 85% |
| Core plus 80 near-neighbor files | no shared-case retrieval/delivery regression |
| Boundary-hardening suite | 100% retrieval and delivery |
| Source indexing | every failure visible; no hidden empty success |

The 60-case limitations suite is intentionally non-gating and non-green at the
complete-contract level. A boundary case should leave the suite only after a
general package improvement changes the product contract.

Pyrrho decisions are recorded but are not counted as Fitz-Sage retrieval
quality. Pyrrho owns its model thresholds, class quality, and context boundary.

## Current Internal Matrix

The accepted matrix records:

| Metric | Result |
|---|---:|
| Required compiled retrieval | 186/192 (96.9%) |
| Required fixed evidence delivery | 186/192 (96.9%) |
| Query-shape recognition | 60/60 (100%) |
| Combined package capability | 246/252 (97.6%) |
| Full contract including exact accepted Pyrrho modes | 193/252 (76.6%) |
| Core retrieval after 80 near-neighbor documents | 20/20 |
| Reload stability | 100% retrieval, delivery, and mode identity |
| Required-suite ingestion | 209/209 files |
| Production gate | pass |

The six remaining required package misses cover grouped code constants,
coordinated prose clauses, one table superlative, one companion service row,
and one mixed table/code expression. They remain visible instead of receiving
case-specific rules.

## Ingestion Evidence

Source-only indexing measurements:

| Corpus | Source size | Indexed | Throughput | Peak RSS |
|---|---:|---:|---:|---:|
| Local core fixtures | 18 files | 18/18 | 60.8 files/s | not retained |
| Local mixed fixtures | 93 files | 93 supported | 51.6 files/s | not retained |
| NapierOne text/code | 54.9 MB | 606/606 | 10.9 files/s | 253 MB |
| NapierOne rich documents | 490.8 MB | 293/303 | 3.4 files/s | 615 MB |
| NapierOne scale | 523.6 MB | 4,994/5,005 | 7.27 files/s | 262 MB |

The ten rich-document failures were eight image-only PDFs and two image-only
PPTX files with no extractable text under the default CPU parser. The eleven
scale failures were ten CSV exports with unusable first-row headers and one
16,383-field row beyond SQLite's column limit.

All NapierOne runs converged to the same manifest and SQLite unit counts after
a forced process exit and resume, with no orphan raw-file records. The 5,005-file
scale index measured a 2.78x SQLite/source storage ratio.

These runs measure parsing, storage, and recovery, not retrieval relevance.
Optional background Qwen time is excluded.

## External Retrieval Evidence

### Broad BEIR

The full pipeline ran over 66,454 NFCorpus, FiQA, and SciFact documents and all
1,271 judged queries:

| Dataset | Plain BM25 nDCG@10 | Full delivered nDCG@10 |
|---|---:|---:|
| NFCorpus | 0.3062 | 0.3367 |
| FiQA | 0.2377 | 0.3179 |
| SciFact | 0.6634 | 0.6170 |

The INT8 reranker produced clear paired gains on NFCorpus and FiQA; its SciFact
effect was inconclusive. Full macro delivered nDCG@10 was `0.4239`.

### Frozen Semantic Holdout

The 240-query ArguAna/Quora holdout measured full macro final nDCG@10 `0.6586`
and delivered nDCG@10 `0.6519`. The current Qwen path did not show a consistent
low-overlap gain on these tasks and added roughly two seconds with reranking.
The holdout remains frozen and is not a tuning set.

### EnterpriseRAG-Bench

The untouched 328-query holdout used 511,961 source files without source
rewriting or document enrichment:

| Variant | Final nDCG@10 | Delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|
| Literal | 0.5276 | 0.5279 | 29.14s |
| Reranker only | 0.5768 | 0.5876 | 39.49s |
| Full | 0.5629 | 0.5780 | 44.20s |

The reranker generalized on aggregate, but multi-document ranking weakened.
Evidence-closure analysis also found many requests for modalities absent from
the physical index. Current code skips those impossible requests and runs valid
closure with request-local terms; matched warm probes measured `13.092s` and
`19.889s`. These probes are latency diagnostics, not a rerun of holdout quality.

## Query Latency

The current matched 60-query SciFact run measured:

| Metric | Result |
|---|---:|
| Mean | 7.43s |
| p50 | 6.77s |
| p95 | 12.56s |
| Queries with delivered relevant evidence | 47/60 |

Managed Qwen, reranking, Pyrrho, and repeated evidence closure dominate query
time; lexical recall is usually the smaller component. Enterprise-scale section
recall and closure remain the most important latency boundary.

Timings are observations from the benchmark machine, not an SLA. Corpus shape,
storage, CPU, cold model loads, and closure obligations materially change them.

## Current Architecture Risks

No feature blocks the hardened release contract, but these measured weaknesses
remain explicit:

1. Pointwise reranking is weak on some multi-document/set questions.
2. Re-pointing hundreds of thousands of unchanged tiny files still walks and
   hashes every file.
3. Managed Qwen expansion has inconclusive aggregate value on the frozen
   external tasks while adding latency.
4. Background Qwen throughput and very large individual documents need broader
   capacity measurements.
5. Pyrrho's current 2,048-token governance context is a separate model boundary.

## Reproduce

```bash
python -m benchmarks.fitz_bench.production_runner \
  --output production-report.json

python -m benchmarks.fitz_bench.external_ingestion_benchmark \
  --profile tiny \
  --workspace .bench_workspace/napierone \
  --output napierone-ingestion.json

python -m benchmarks.fitz_bench.beir_benchmark \
  --offline \
  --reuse-workspace \
  --resume-queries \
  --output benchmarks/results/beir_full.json
```

The canonical measurements, component ablations, exact revisions, and report
links are in [Benchmarks](BENCHMARK.md).
