# Production Readiness

fitz-sage's default production contract is:

> Point fitz-sage at a folder of reasonably clean, supported company
> documents and retrieve the correct source evidence for ordinary business
> questions without project-specific retrieval code.

This is narrower than "arbitrary files work." It is also stronger than a
collection of isolated retrieval unit tests.

## Package Responsibilities

fitz-sage owns:

- recursive discovery of supported files
- explicit reporting of unsupported and failed files
- parsing supported formats into searchable evidence units
- deterministic and managed semantic query terms
- BM25 recall, reranking, evidence reading, and evidence compilation
- temporal, comparison, aggregation, and other general query-shape handling
- mechanical transport and recording of exact Pyrrho planning and governance
  output
- evidence provenance and inspectable retrieval-run records

## User Responsibilities

The user owns:

- OCR/parser selection for documents the default parser cannot read
- raw-log compression or rewriting
- domain-specific cleanup
- deciding that differently written identifiers are equivalent
- acronym, synonym, and vocabulary mappings specific to the corpus
- removing secrets and documents that should not be indexed
- choosing supported extensions and optional parser dependencies

fitz-sage does not silently normalize `ATX-123`, `ATX_123`, and `ATX 123`.
It does not infer that two organization-specific abbreviations are aliases.

## Production Matrix

`python -m benchmarks.fitz_bench.production_runner` runs:

1. Three independent standard and holdout corpora.
2. The core corpus with 80 deterministic near-neighbor documents.
3. The grown corpus again after loading it in a fresh engine.
4. Real PDF, DOCX, PPTX, XLSX, SQL, Go, Java, and TypeScript files.
5. Sixty explicit query-shape cases with positive and negative controls.
6. An 11-case required boundary-hardening suite.
7. The intentionally non-green 60-case limitations suite.

The matrix starts at folders and files. It does not inject prebuilt chunks or
retrieval results.

Every case reports compiled retrieval, fixed evidence delivery, query shape,
and Pyrrho outcomes separately. Cases without an assertion for one of
those metrics are excluded from that metric's denominator.
Missing evidence is attributed to the earliest observable failing stage:

- recall
- reranking
- final selection
- evidence reading or compilation
- evidence content/granularity
- fixed evidence delivery and Pyrrho decision

## Ingestion Health

Every discovered file reaches one visible state. Supported-file failures carry
the file path, stage, and error. Unsupported files are listed separately.

`complete` means every supported file indexed successfully. `query_ready`
means no supported file is still pending; it may still be true for a partial
collection with explicitly reported failures.

This distinction allows useful partial retrieval without representing a
partial collection as healthy.

## Release Interpretation

Required retrieval suites gate on at least 85% compiled retrieval
success. Query-shape recognition has its own 85% gate. Any retrieval or
delivery regression on shared core cases after adding 80 near-neighbor
documents fails the production gate even if the aggregate remains above 85%.
These are minimum contracts, not a claim that every domain or document works.

The optional-format and full limitations suites remain non-gating. The required
boundary-hardening suite gates its 11 retrieval contracts at 100%. Limitation
failures are retained as product-boundary evidence and should only be removed
when a general package improvement changes the boundary.

Pyrrho quality is not inferred from retrieval cases. Its accepted default was
trained with benchmark-derived deterministic rows and requires a future
independent evaluation for generalization claims. Retrieval reports still
preserve its exact decisions so integration failures remain visible.

## Measured Baseline

The final 2026-07-27 local matrix ran all ten suites from fresh folders and
fresh isolated workspaces:

| Metric | Result |
|--------|--------|
| Required retrieval | 186/192 (96.9%) |
| Required governed evidence delivery | 186/192 (96.9%) |
| Query-shape recognition | 60/60 (100%) |
| Combined package capability | 246/252 (97.6%) |
| Full contract, including diagnostic Pyrrho modes | 193/252 (76.6%) |
| Noisy core retrieval after 80 added documents | 20/20 |
| Noisy core reload stability | 100% retrieval, delivery, and mode identity |
| Required-suite ingestion | 209/209 files |
| Production gate | pass |

Suite-level package results:

| Suite | Retrieval or shape | Delivery | Purpose |
|-------|--------------------|----------|---------|
| Core | 20/20 | 20/20 | baseline behavior |
| Holdout | 47/50 | 47/50 | unseen corpus |
| Holdout 2 | 47/50 | 47/50 | second unseen corpus |
| Core + noise | 20/20 | 20/20 | 80 near-neighbor documents |
| Query shapes | 60/60 | n/a | temporal, comparison, aggregation, narrow |
| PDF/DOCX/PPTX | 24/24 | 24/24 | rich-document facts |
| SQL/Go/Java/TypeScript/PPTX | 17/17 | 17/17 | base and code formats |
| XLSX (optional) | 5/5 | 5/5 | optional parser path |
| Hardened boundaries | 11/11 | 11/11 | required long-document, bridge, precision, and structured cases |
| Limitations (non-gating) | 52/52 | 52/52 | package boundary plus diagnostic Pyrrho outcomes |

The full-contract rate must not be described as retrieval quality. Pyrrho's
exact decisions are useful integration evidence, but its current benchmark
results are not independent model-quality evidence.

## Current Boundaries

The matrix does not support a claim that arbitrary company folders work
without preparation. The measured contract excludes:

- raw-log compression, rewriting, OCR recovery, and domain cleanup
- silent equivalence between differently written identifiers
- guaranteed expansion of private abbreviations or synonyms
- unlimited understanding of long, poorly segmented prose
- guaranteed completion of unrelated multi-intent requests without an
  explicit bridge
- perfect semantic filtering or aggregation over every table shape

See [Limitations](../LIMITATIONS.md) for the case-level boundary record.

## Performance Evidence

The matrix took 2,455.3 seconds on the benchmark machine. The 98-file noisy
corpus took 335.5 seconds to index, or about 0.29 files/second. Required-suite
queries averaged 4.1 seconds with a 3.6-second median, including cold starts;
the slowest limitation query took 27.9 seconds.

These figures are observations, not an SLA. They show that enrichment of many
small files and broad retrieval over pathological documents need production
capacity testing on the user's hardware and corpus.

## Reproduce

```bash
python -m benchmarks.fitz_bench.production_runner \
  --output production-report.json
```

Generated JSON and Markdown reports are ignored by Git because they contain
machine-specific paths and model fingerprints. Preserve the report produced by
the exact release candidate in external release evidence.
