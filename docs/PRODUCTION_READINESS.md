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

The historical matrix took 2,455.3 seconds on the benchmark machine. Its
335.5-second figure for a 98-file noisy corpus included model-backed
keyword/entity/hierarchy work and is not source-index throughput.

With source indexing separated from optional enrichment, the 2026-07-28 CPU
benchmark made 18 core files query-ready in a median 0.296 seconds across three
cold runs (60.8 files/second), and 93 mixed files query-ready in 1.803 seconds
(51.6 files/second). Both runs had zero source-index failures; the mixed run
reported one XLSX as unsupported under the default CPU parser.

The source-only core run passed 20/20 retrieval, delivery, and package-capability
contracts with 100% required recall and no forbidden evidence. It passed 14/20
complete contracts; all six failures were attributed to accepted Pyrrho
outputs, not missing source retrieval.

The 2026-07-28 external ingestion run used unchanged, SHA-256-verified files
from the public NapierOne dataset:

| Slice | Source size | Indexed | Failures | Throughput | Peak RSS |
|---|---:|---:|---:|---:|---:|
| CSV/TXT/JSON/JavaScript/HTML/XML | 54.9 MB | 606/606 | 0 | 10.9 files/s | 253 MB |
| PDF/DOCX/PPTX | 490.8 MB | 293/303 | 10 | 3.4 files/s | 615 MB |
| CSV/JSON/JavaScript/HTML/XML scale | 523.6 MB | 4,994/5,005 | 11 | 7.27 files/s | 262 MB |

All 100 DOCX files indexed. Eight PDFs had no embedded text and two PPTX files
had no text shapes, so the default CPU parsers explicitly rejected them instead
of recording empty searchable documents. That is a 3.3% supported-file failure
rate for the rich slice and demonstrates the documented OCR/image-content
boundary.

The 5,005-file scale slice rejected ten CSV exports whose first rows did not
provide usable headers and one CSV whose 16,383-field first row exceeded
SQLite's column limit. The other 4,000 JSON, JavaScript, HTML, and XML files all
indexed. Its 515.6 MB of indexed source produced a 1.43 GB SQLite database, a
2.78x storage ratio that should be included in capacity planning.

All three slices exceeded the one-file-per-second target and left their SQLite
counts unchanged after an immediate re-point. An abrupt process exit followed
by a resume produced the same manifest inventory and SQLite retrieval-unit
counts as the clean run, with no orphan raw-file records. The tiny slices
crashed after ten durable files; the 5,005-file scale slice crashed after 100.

The later Quora retrieval holdout exposed a different scale boundary:
ordinary no-change `point()` did not complete within 20 minutes over 522,931
tiny projected files because it still walks and hashes source files. A strict
benchmark-only path validated the persisted index and every persisted content
hash against the deterministic adapter mapping in about 5.5 seconds. Public
persisted-collection reuse remains future work; the benchmark shortcut is not
product behavior.

Required-suite queries averaged 4.1 seconds with a 3.6-second median, including
cold starts; the slowest limitation query took 27.9 seconds.

These figures are observations, not an SLA. They show that enrichment of many
small files and broad retrieval over pathological documents need production
capacity testing on the user's hardware and corpus. The 909-file tiny slices
and 5,005-file scale slice are parser and recovery samples, not a
retrieval-quality evaluation and not a guarantee over NapierOne's full
500,000-file dataset.

Real-file ingestion and labeled retrieval are separate claims. NapierOne
measures parsing, storage, throughput, and recovery over unchanged files. The
external BEIR run measures whether judged relevant documents survive recall,
reranking, final selection, compilation, and delivery across biomedical,
financial, and scientific corpora.

## External Retrieval Evidence

The 2026-07-30 source-only BEIR run used 66,454 external documents and all
1,271 judged test queries from NFCorpus, FiQA, and SciFact. It ran from clean
commit `2893be4f` and made no per-dataset retrieval changes.

| Dataset | Plain BM25 nDCG@10 | Full recall nDCG@10 | Final nDCG@10 | Delivered nDCG@10 |
|---|---:|---:|---:|---:|
| NFCorpus | 0.3062 | 0.3264 | 0.3377 | 0.3367 |
| FiQA | 0.2377 | 0.2445 | 0.3188 | 0.3179 |
| SciFact | 0.6634 | 0.6403 | 0.6529 | 0.6170 |

The literal Fitz-Sage path, with Qwen and cross-encoder scoring disabled,
exceeded the local whole-document BM25 Recall@50 on all three datasets: 0.2164
versus 0.2101 on NFCorpus, 0.4746 versus 0.4459 on FiQA, and 0.8952 versus
0.8704 on SciFact. Typed indexing, deterministic query planning, and retrieval
fusion therefore provide measurable recall value without managed semantic
terms.

A paired four-variant ablation isolated the managed components:

| Variant | Macro final nDCG@10 | Macro delivered nDCG@10 | Mean latency |
|---|---:|---:|---:|
| Literal path | 0.4042 | 0.3902 | 2.07s |
| Literal plus Qwen | 0.4054 | 0.3916 | 4.03s |
| Literal plus reranker | 0.4372 | 0.4244 | 4.18s |
| Full pipeline | 0.4365 | 0.4239 | 6.15s |

The reranker produced clear paired quality gains on NFCorpus and FiQA for
about 2.10 seconds per query. Qwen produced a clear recall-ordering gain only
on NFCorpus. With reranking active, it added 1.98 seconds and changed macro
final nDCG@10 by -0.0008.

The follow-up frozen ArguAna/Quora holdout is now complete. Without reranking,
Qwen reduced two-dataset macro recall nDCG@10 by 0.0106 and final nDCG@10 by
0.0072 while adding 2.44 seconds. Its Quora regressions were conclusive, and
no consistent low-overlap gain appeared. With reranking active, Qwen changed
macro final nDCG@10 by only +0.0022 while adding 2.06 seconds. The current
managed expansion path is therefore a measured limitation on those BEIR tasks,
not a guaranteed recall improvement.

This does not justify removing Qwen from the package. BM25 remains lexical,
and the holdout is not an application-shaped company-document benchmark.
Qwen retains a valid role as the best-effort general-language bridge between a
user's wording and source vocabulary. See the
[semantic holdout report](evaluation/beir-semantic-holdout-2026-07-30.md).

The post-fix compiler no longer has the capitalization-derived phrase-anchor
failure measured in the earlier run. Thirteen SciFact queries still lost their
last judged-relevant candidate during compilation, and all 13 contained a
literal structured identifier. Fitz-Sage intentionally does not guess
equivalence between identifier variants. Corpus cleanup or a user-owned
preprocessing mapping owns that relationship. A public vocabulary hook remains
deferred.

Delivered rankings exactly matched compiled rankings in all three datasets.
Pyrrho verdicts were recorded unchanged but were not scored as relevance
labels and did not gate retrieval.

FiQA contained 38 records with no title or text, including one judged-relevant
record. Fitz-Sage reported all 38 as unsearchable; the adapter did not hide
them. All non-empty FiQA records and every NFCorpus and SciFact record indexed.

Mean canonical query latency was 5.98 seconds for NFCorpus, 5.58 seconds for
FiQA, and 6.90 seconds for SciFact on the benchmark machine. The complete
four-variant run took about 6.3 hours. Managed Qwen expansion, cross-encoder
scoring, and the exact Pyrrho evidence decision dominate query time; lexical
recall itself averaged roughly 0.16 to 0.38 seconds.

See the
[full component-ablation record](evaluation/beir-component-ablation-2026-07-30.md)
and
[semantic holdout](evaluation/beir-semantic-holdout-2026-07-30.md)
for paired confidence intervals, timing details, and reproduction instructions.

## Reproduce

```bash
python -m benchmarks.fitz_bench.production_runner \
  --output production-report.json
```

Run the external real-file ingestion and hard-crash benchmark separately:

```bash
python -m benchmarks.fitz_bench.external_ingestion_benchmark \
  --profile tiny \
  --workspace .bench_workspace/napierone \
  --output napierone-ingestion.json
```

Run the independent labeled retrieval evaluation:

```bash
python -m benchmarks.fitz_bench.beir_benchmark \
  --offline \
  --reuse-workspace \
  --resume-queries \
  --output benchmarks/results/beir_full.json
```

Generated JSON and Markdown reports are ignored by Git because they contain
machine-specific paths and model fingerprints. Preserve the report produced by
the exact release candidate in external release evidence.
