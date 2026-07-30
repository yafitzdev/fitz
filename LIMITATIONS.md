# Limitations

fitz-sage is a retrieval package. It tries to return grounded evidence from the
documents it was given. It does not try to clean, rewrite, or normalize a user's
domain data by default.

## Input Contract

fitz-sage works best when source material already uses the identifiers,
abbreviations, and terminology that users will query.

Users are responsible for domain-specific preprocessing such as:

- normalizing IDs across logs, reports, and tables
- expanding private abbreviations
- compressing or summarizing raw logs before ingestion
- deciding whether two identifier spellings are equivalent
- providing authority markers when current/final status is not expressed in the
  source text

When data is inconsistent, fitz-sage should prefer returning insufficient or
disputed evidence over silently inventing equivalence.

## Identifier Matching

Identifier matching is literal by default.

Examples that are not treated as equivalent unless the corpus says so:

- `AX-156` vs `AX_156`
- `AX-156` vs `AX 156`
- `AX-156` vs `AX156`
- `MOD_88X` vs `MOD-88X`

If those forms are aliases in a user's domain, the user should normalize them
before ingestion or include explicit alias evidence in the corpus.

Sparse retrieval may still surface lexically similar neighbors in the evidence
pool. That is not an equivalence guarantee. The safe behavior is to return
insufficient or disputed evidence unless the exact queried form or an explicit
alias is present.

## Abbreviations And Aliases

fitz-sage may use explicit in-corpus definitions, for example:

`CBT means Cell Balancing Task`

It should not generate private abbreviation expansions on its own. If an
abbreviation is not defined in the corpus, retrieval may miss the intended
document or return insufficient evidence.

## Semantic Keywords And Equivalence

Managed Qwen proposes semantic keywords for the BM25 recall stage. These are
best-effort search suggestions, not deterministic mappings or proof that two
terms are equivalent.

fitz-sage does not include a fixed synonym/acronym dictionary. The following
relationships are therefore not package guarantees:

- `fetch` means `retrieve`
- `db` means `database`
- `authorisation` means `authorization`
- `waive` means `waiver`

Qwen may still suggest one of those related terms and surface useful evidence.
Users who require consistent domain equivalence must provide it through data or
query preprocessing; a public mapping hook is deferred.

The original query remains a recall leg, but all recall legs share a bounded
candidate pool. A poor semantic suggestion can therefore change ordering or
displace a useful literal candidate. Semantic expansion is not guaranteed to
be neutral when it fails to help.

## Exact Literal Anchors

Fitz-Sage uses recognized exact identifiers to keep a narrow query tied to
concrete evidence. Capitalization alone does not create a hard anchor, so an
ordinary title-cased question cannot filter out otherwise relevant evidence.

Exact identifier anchoring remains intentional. If a query says `ATX-123`,
Fitz-Sage does not silently accept `ATX_123`. Scientific notation such as
`B12` versus `B-12` can therefore expose the same literal-input contract; the
user owns normalization when those forms are equivalent in their corpus.

## Temporal And Version Scope

fitz-sage should not infer chronological authority from version-looking strings
alone. For example, `REL-2026.02` is not automatically more authoritative than
`REL-2025.09` unless the surrounding evidence says `current`, `final`, `latest`,
or otherwise establishes the scope.

Known hard cases:

- mixed historical/current facts in one source
- stale documents next to newer code or tables
- final-vs-draft language without clear authority markers
- version identifiers that look like dates but are not dates

Temporal routing can order current/final evidence ahead of historical evidence,
but fitz-sage does not delete the historical source from the evidence pack.
The time at which a file was indexed is not treated as document recency.
Pyrrho receives the raw retrieved sources and decides whether they are
sufficient or disputed.

## Tables

Literal row lookup is supported for exact identifiers. Clear requests for
records with a property or status can also trigger a bounded structured-row
scan. fitz-sage does not rewrite row keys to find separator variants by
default.

Known hard cases:

- row keys with inconsistent formatting across files
- superlatives over noisy tables
- comparison queries that require multiple rows and prose context
- large tables beyond the bounded scan when the queried key is absent or
  misspelled

## Long Documents

Reranking uses a bounded query-centered excerpt while selected evidence retains
the original source text. This prevents a late literal fact from being hidden
solely because a document is long or headingless, but it is not an unlimited
long-context guarantee. Poorly segmented documents can still crowd several
relevant regions into one candidate, and Pyrrho currently evaluates at most
2,048 tokens.

## Conflicts

fitz-sage can return disputed evidence when multiple returned sources disagree.
Current limitations remain around:

- PASS/FAIL style contradictions
- stale docs versus code behavior
- multiple owners/statuses in different sources
- conflicts where the decisive source is outside the delivered evidence set

Fitz-Sage returns Pyrrho's verdict without local confidence thresholds,
evidence floors, or dispute-stability overrides.
False sufficient, insufficient, or disputed decisions are Pyrrho model debt and
should be fixed by retraining Pyrrho rather than by adding hidden Fitz-Sage
heuristics.

## Raw Logs

Raw logs are not a target input format. Users should compress, summarize, or
otherwise structure logs before ingestion. fitz-sage should not own that cleanup
step.

## Very Large File Counts

Source indexing and no-change detection currently operate on individual files.
Calling `point()` again walks and hashes those files even when the index is
already query-ready.

In a 2026-07-30 diagnostic over 522,931 tiny projected Quora documents, an
ordinary no-change `point()` did not complete within 20 minutes. A
benchmark-only strict reuse path validated the persisted index, source-ID
mapping, and every persisted content hash against the deterministic adapter
mapping in about 5.5 seconds, but that path does not change the public product
behavior.

This is an extreme-file-count startup limitation, not a query-latency
limitation. Avoid repeatedly pointing unchanged collections at folders
containing hundreds of thousands of tiny files. A future product change should
address persisted collection reuse explicitly rather than weakening source
identity checks.

## Current Limitation Benchmark

The limitations benchmark intentionally contains cases that should fail or
surface product boundaries. It is not a marketing benchmark.

The 2026-07-27 hardening run measured 60 limitation cases. All 52 cases with
evidence assertions passed retrieval and fixed evidence delivery (100%),
required recall was 100%, and no forbidden evidence was returned. The three
former package misses now pass through general mechanisms:

- query-centered reranking excerpts for facts late in long documents
- a query-bound bridge for explicit definitions such as `QRS means Queue
  Recovery Service`
- a bounded scan for clearly structured record/property requests

These mechanisms preserve raw source text. They do not normalize identifiers,
invent expansions, or create a persistent synonym dictionary.

The same run passed 35 of 60 complete contracts. All 25 failures were attributed
to Pyrrho verdicts or failure modes, while retrieval and delivery still passed.
The run used the accepted immutable `pyrrho-v2-nano-g1` default at its current
2,048-token contract. Pyrrho's training data included benchmark-derived
deterministic rows, so the 35/60 result is diagnostic integration evidence, not
an independent model-quality claim. Improving those 25 decisions belongs in
Pyrrho training and release work, not in Fitz-Sage policy code.

The required production suites measured:

- 186/192 retrieval contracts (96.9%)
- 186/192 governed evidence-delivery contracts (96.9%)
- 60/60 query-shape contracts (100%)
- 246/252 combined package capability contracts (97.6%)
- 20/20 core retrieval contracts after adding 80 near-neighbor documents
- 100% retrieval, delivery, and governance identity stability after reload

The remaining six required-suite package misses cover:

- a grouped code constant whose individual deployment environment values were
  not compiled into the required symbol evidence
- two coordinated prose requests where the second clause was absent from
  compiled evidence
- a table superlative whose winning row was absent from compiled evidence
- a mixed service-owner request whose companion service row was not recalled
- a mixed table/code request whose scheduler expression was absent from
  compiled evidence

All 60 query-shape controls pass. These misses remain visible because the
benchmark is a boundary record, not a target to fit case by case.

Performance is measured, not guaranteed. The former 335.5-second figure for a
98-file noisy corpus included model-backed keyword/entity/hierarchy work and
therefore did not measure source indexing alone.

After separating the lifecycle, the 2026-07-28 CPU benchmark measured:

- 18 core files query-ready in a median 0.296 seconds across three cold runs
  (60.8 files/second, zero indexing failures)
- 93 mixed benchmark files query-ready in 1.803 seconds
  (51.6 files/second, zero indexing failures, one unsupported XLSX under the
  default CPU parser)
- unchanged 18-file re-pointing in roughly 0.02-0.03 seconds

The source-only core retrieval run passed 20/20 retrieval, evidence-delivery,
and package-capability contracts with 100% required recall, without entity or
hierarchy enrichment. It passed 14/20 complete contracts; all six remaining
failures were attributed to the accepted Pyrrho outputs.

These fixture corpora contain many small files and are not a large-document
throughput guarantee. Rich parsers, very large documents, slow storage, and
background Qwen enrichment need separate capacity tests. Required-suite queries
previously averaged 4.1 seconds with a 3.6-second median; the slowest limitation
query took 27.9 seconds.

The external NapierOne benchmark adds real-file ingestion evidence without
checking corpus files into Fitz-Sage. The measured tiny selections contained
909 files and 545.7 MB of source data across CSV, TXT, JSON, JavaScript, HTML,
XML, PDF, DOCX, and PPTX.

- The text/code slice indexed 606/606 files at 10.9 files/second.
- The rich-document slice indexed 293/303 files at 3.4 files/second.
- Eight image-only PDFs and two image-only PPTX files had no text available to
  the default CPU parser and were reported as failures.
- Both slices resumed from a hard process exit and converged exactly to their
  clean-run manifest and SQLite counts.

The separate NapierOne `small` scale run measured 5,005 files and 523.6 MB
across CSV, JSON, JavaScript, HTML, and XML. It indexed 4,994 files at 7.27
files/second, left its index unchanged on re-point, and converged exactly after
a hard exit at file 100.

All 4,000 JSON, JavaScript, HTML, and XML files indexed. Eleven CSV files did
not: ten spreadsheet exports placed blank/title rows before the real table
header, and one had 16,383 fields in its first row, beyond SQLite's 2,000-column
limit. Fitz-Sage does not guess which later row should become the schema or
silently split an ultra-wide table. Cleaning or reshaping those files belongs
to the user. The resulting SQLite database was 1.43 GB for 515.6 MB of indexed
source, a measured 2.78x storage ratio.

This does not prove that every NapierOne file, every supported format, or an
arbitrary company folder will work. The full upstream corpus contains more than
500,000 files; only selected 100- and 1,000-file-per-type archives were
measured. The benchmark does not assess whether retrieved evidence answers
domain questions, and it does not include optional Qwen enrichment time.
Image-only documents require a user-selected OCR-capable parser.

The 2026-07-30 external BEIR retrieval run adds independent relevance judgments
across 66,454 biomedical, financial, and scientific documents. The literal
Fitz-Sage path, with managed Qwen expansion and cross-encoder scoring disabled,
exceeded the benchmark-local whole-document BM25 Recall@50 on all three
datasets:

- NFCorpus: 0.2164 versus 0.2101
- FiQA: 0.4746 versus 0.4459
- SciFact: 0.8952 versus 0.8704

With the full canonical pipeline, delivered nDCG@10 was:

- NFCorpus: 0.3367 versus the plain-BM25 baseline's 0.3062
- FiQA: 0.3179 versus 0.2377
- SciFact: 0.6170 versus 0.6634

This means the central typed BM25 recall loop generalizes beyond the internal
fixtures without relying on Qwen. The INT8 reranker produced clear paired
quality gains on NFCorpus and FiQA, while its SciFact gain was inconclusive.

Managed Qwen expansion cost about 1.98 seconds per query with reranking active.
It produced a conclusive recall-ordering gain only on NFCorpus and changed the
three-dataset macro final nDCG@10 by -0.0008. The broad suite did not directly
target ordinary synonym and paraphrase mismatch, so a separate frozen holdout
was created before making a product decision.

That 240-query ArguAna/Quora holdout did not validate the current managed
Qwen path. Without reranking, expansion reduced macro recall nDCG@10 by 0.0106,
reduced final nDCG@10 by 0.0072, and added 2.44 seconds per query. The Quora
recall and final regressions were conclusive. With reranking active, expansion
changed macro final nDCG@10 by only +0.0022 while adding 2.06 seconds, and the
per-dataset effects were inconclusive. Low-overlap queries did not show a
consistent expansion gain.

The current managed model is therefore not a reliable semantic bridge, and the
measured evidence does not justify its always-on cost. This is a limitation of
the current model, prompt, and fixed-budget fusion behavior, not proof that all
semantic query expansion is useless. The holdout remains frozen and will not
be used to tune those components. See the
[semantic holdout report](docs/evaluation/beir-semantic-holdout-2026-07-30.md)
for the complete method, paired intervals, and operational findings.

Thirteen SciFact queries had a judged-relevant final candidate but lost all
judged-relevant evidence during compilation. Every query contained a literal
structured identifier, including examples such as `anti-interleukin-2`,
`SHP-2`, `CK-666`, and `FOXO3`. Fitz-Sage does not silently treat spelling,
separator, abbreviation, or naming variants as equivalent. Users must provide
consistent source/query text or apply their own preprocessing mapping. A
public vocabulary hook is deferred and is not part of the current API.

All 1,271 judged queries completed. All NFCorpus and SciFact documents indexed.
FiQA contained 38 upstream records with empty title and text fields, including
one judged-relevant record; those were reported as unsearchable input instead
of being hidden or treated as successful indexing.

Use this file as the public contract. Use the benchmark to decide which
limitations are worth removing later.
