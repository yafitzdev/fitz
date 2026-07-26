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
Pyrrho receives the raw retrieved sources and decides whether they are
sufficient or disputed.

## Tables

Literal row lookup is supported for exact identifiers. fitz-sage does not
rewrite row keys to find separator variants by default.

Known hard cases:

- row keys with inconsistent formatting across files
- superlatives over noisy tables
- comparison queries that require multiple rows and prose context
- large tables when the queried key is absent or misspelled

## Conflicts

fitz-sage can return disputed evidence when multiple returned sources disagree.
Current limitations remain around:

- PASS/FAIL style contradictions
- stale docs versus code behavior
- multiple owners/statuses in different sources
- conflicts where the decisive source is outside the governed evidence prefix

Once the query-shape evidence floor is present, fitz-sage returns Pyrrho's
verdict without local confidence thresholds or dispute-stability overrides.
False sufficient, insufficient, or disputed decisions are Pyrrho model debt and
should be fixed by retraining Pyrrho rather than by adding hidden Fitz-Sage
heuristics.

## Raw Logs

Raw logs are not a target input format. Users should compress, summarize, or
otherwise structure logs before ingestion. fitz-sage should not own that cleanup
step.

## Current Limitation Benchmark

The limitations benchmark intentionally contains cases that should fail or
surface product boundaries. It is not a marketing benchmark.

The 2026-07-27 production matrix measured 60 limitation cases. Of the 52 cases
with evidence assertions, 48 passed retrieval and governed delivery (92.3%).
No forbidden evidence was returned by this suite. The four package capability
misses were:

- two final facts buried in long headingless or single-paragraph documents
  dropped during reranking
- one explicit private-abbreviation bridge whose companion source was not
  recovered during evidence reading/compilation
- one filtered table lookup whose target row was missed during recall

These failures stay in the suite. They should only turn green after a general
retrieval improvement, not a case-specific synonym or identifier rule.

The same run passed 32 of 60 complete governed contracts. Pyrrho disagreed with
the expected mode on 28 cases, including some of the four package misses. This
run used an explicit local diagnostic Pyrrho artifact with no package manifest
and a failing parity report. Its exact decisions are preserved for integration
diagnosis, but its mode accuracy is not release-grade governance evidence.

The required production suites measured:

- 175/181 retrieval contracts (96.7%)
- 175/181 governed evidence-delivery contracts (96.7%)
- 58/60 query-shape contracts (96.7%)
- 233/241 combined package capability contracts (96.7%)
- 20/20 core retrieval contracts after adding 80 near-neighbor documents
- 100% retrieval, delivery, and governance identity stability after reload

The remaining required-suite package misses cover:

- a grouped code constant whose individual environment-variable values were
  not compiled into the required symbol evidence
- a multi-intent question whose two unrelated source requests had no explicit
  bridge
- one latest-state case where stale evidence remained in the ranking
- three mixed table/code cases where a required companion row or code fragment
  was absent from compiled evidence

Two of 60 query-shape controls also remain red: one narrow rotation-duration
query was classified as comparative, and one narrow route-path query was
classified as aggregative.

Performance is measured, not guaranteed. On the benchmark machine, indexing
the 98-file noisy corpus took 323.5 seconds (0.30 files/second), and ordinary
suite queries averaged roughly 3-4 seconds. A pathological limitation query
took 50 seconds. Many small enriched files and unusually broad/long evidence
can therefore be operational bottlenecks even when retrieval is correct.

Use this file as the public contract. Use the benchmark to decide which
limitations are worth removing later.
