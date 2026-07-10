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
- marking which stale/current source should be authoritative

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

As of the 2026-07-09 local run, 34 of 60 cases pass. Mean required-evidence
recall is 97.5%, and no case returns evidence marked forbidden by the current
contract. The 26 red cases split into:

- 2 retrieval/closure misses: a late fact in a long headingless document and an
  explicit abbreviation bridge whose target source was not recovered
- 24 cases where required evidence was present but Pyrrho returned a different
  mode than expected

The most important limitation categories are:

- abbreviation / cross-source bridge handling
- strict identifier precision and separator variants
- structured table filtering and comparison
- conflict governance
- stale/current evidence arbitration

Companion regression suites from the same run:

- core: 17/20 passed, 100% mean required-evidence recall, 0 forbidden hits
- holdout: 36/50 passed, 96.3% mean required-evidence recall, 0 forbidden hits

The three core failures are Pyrrho mode errors on complete evidence. Holdout has
three retrieval/closure misses and eleven Pyrrho mode errors.

Use this file as the public contract. Use the benchmark to decide which
limitations are worth removing later.
