# Limitations

Fitz-Sage is a retrieval package for reasonably clean, supported documents. It
tries to return grounded source evidence; it does not silently repair or
reinterpret a user's domain data.

## Responsibility Boundary

| Fitz-Sage owns | The user owns |
|---|---|
| Supported-file discovery and visible failures | Choosing and preparing the corpus |
| Parsing into sections, symbols, and native tables | Raw-log compression or rewriting |
| Literal and best-effort semantic query recall | Private acronym, synonym, and ID mappings |
| General query shapes such as temporal, comparison, and aggregation | Deciding which identifier forms are equivalent |
| Reranking, source reading, evidence compilation, and provenance | OCR/parser selection for image-only or complex inputs |
| Exact transport of Pyrrho planning and governance output | Removing secrets and documents that must not be indexed |

The boundary is intentionally explicit. Domain-specific cleanup differs by
company and corpus; hiding it inside a universal package would create false
matches that are difficult to audit.

## Identifier Matching

Identifier equivalence is literal by contract. Fitz-Sage does not declare these
forms equivalent:

- `AX-156`, `AX_156`, `AX 156`, and `AX156`;
- `MOD_88X` and `MOD-88X`;
- a private abbreviation and its undocumented expansion.

FTS tokenization can still make variants share lexical pieces and therefore
enter the same broad recall pool. Exact identifier anchoring prevents that
lexical overlap from becoming a package-level equivalence claim.

If two forms are aliases in a user's domain, normalize them before ingestion or
put explicit alias evidence in the corpus. There is no public mapping hook.

## Semantic Query Terms

Managed Qwen proposes words and short phrases that may occur in relevant
source. These are best-effort recall suggestions, not a dictionary and not
proof that two terms mean the same thing.

The original query remains a recall leg, but literal and expanded legs share a
bounded pool. Candidate competition is intentional broad-recall behavior:
alternate vocabulary needs room to enter, so a fixed-cutoff score may improve
or decline on an individual query.

If Qwen returns malformed output or fails during a query, the failure is traced
and retrieval continues with the literal prepared plan. Background Qwen failure
is reported separately and does not invalidate the source index.

The frozen ArguAna/Quora holdout did not show a consistent low-overlap gain from
the current managed expansion path. With reranking active, Qwen changed macro
final nDCG@10 by `+0.0022` while adding `2.06s` per query; per-dataset effects
were inconclusive. That is a measured limitation of the current model and
prompt on those tasks, not evidence that lexical retrieval no longer needs a
general-language bridge.

## Temporal Scope And Authority

Fitz-Sage recognizes time-related query shape and can prefer evidence whose
content explicitly says `current`, `final`, `latest`, or names the requested
period. It does not infer authority from:

- ingestion time;
- filesystem modification time;
- SQLite row age;
- a version-looking token by itself.

For example, `REL-2026.02` is not automatically more authoritative than
`REL-2025.09`. The source must express chronology or status when it matters.
Historical evidence remains available so Pyrrho can see conflicts.

Hard cases include mixed historical/current facts in one section, draft and
final documents without explicit labels, and version identifiers that resemble
dates but are not chronological.

## Native Tables

The native structured path covers configured delimited files, `.csv` and `.tsv`
by default. Embedded document tables remain section text, and XLSX is not a
native default format.

Fitz-Sage supports table-name/column recall, row-value BM25, exact identifier
lookup, and bounded deterministic filter/sort plans. Current limits include:

- no automatic repair when blank/title rows precede the actual CSV header;
- no silent splitting of ultra-wide rows beyond SQLite's column limit;
- no identifier separator normalization;
- bounded fallback scans when no indexed value or supported predicate points to
  the row;
- no generated multi-table join orchestration;
- finite result limits for large sets.

Complex schema repair and reshaping remain user-owned.

## Long Documents

Reranking uses bounded source-faithful candidate text and can select a
query-centered excerpt for long documents. Delivered evidence retains the
selected original source content. This prevents a late literal fact from being
hidden solely by its position, but it is not unlimited long-context reasoning.

Poorly segmented documents can still put several relevant regions into one
candidate. The reranker accepts 512 tokens per pair, and the accepted Pyrrho
model currently accepts at most 2,048 tokens for its governance decision.

## Multi-Document And Set Coverage

The current cross-encoder scores each candidate independently. It does not
optimize a result set for diversity or joint coverage.

On the frozen EnterpriseRAG-Bench holdout, reranker-only final nDCG@10 improved
from `0.5192` to `0.5892` on 265 single-document questions but declined from
`0.5633` to `0.5247` on 63 multi-document questions. This is a Fitz-Sage
architecture limitation and points toward set-aware coverage after pointwise
reranking.

Comparison, aggregation, and evidence closure reduce some one-sided failures,
but finite recall, rerank, read, and delivery budgets remain. A list/count pack
is not proof of exhaustive corpus coverage.

## Conflicts And Governance

Pyrrho owns every evidence verdict. Fitz-Sage starts with the first three ranked
sources, adds two only after exact `INSUFFICIENT`, and returns the stopping
prefix without local confidence thresholds, query-shape evidence floors, or
dispute overrides.

False sufficient, insufficient, or disputed decisions are Pyrrho model debt.
They should be addressed in Pyrrho rather than hidden by Fitz-Sage policy code.
Conflicts can also be missed when Pyrrho returns a false terminal verdict before
the decisive source enters the prefix, or when that source lies beyond the
delivery cap.

## Raw Logs, Scans, And Unsearchable Inputs

Raw logs are not a target input contract. Users should compress, summarize, or
structure them before ingestion. Fitz-Sage does not own that cleanup step.

The default rich-document parsers require extractable text. Image-only PDFs or
PPTX files can fail visibly instead of becoming empty successful documents; use
an explicitly selected OCR/vision-capable parser when needed. Empty upstream
records are likewise reported as unsearchable.

## Extreme File Counts

`point()` walks and hashes individual files to validate source identity. An
ordinary unchanged re-point over 522,931 tiny projected Quora documents did not
finish inside a 20-minute diagnostic window. A benchmark-only persisted-index
validation completed in about 5.5 seconds, but that shortcut is not public
product behavior.

Avoid repeatedly pointing unchanged collections at folders containing hundreds
of thousands of tiny files. Persisted collections can be queried by name
without re-pointing, but there is no fast validated no-change re-point that
skips the full source walk and hashing pass.

## Current Measured Boundaries

These values are observations, not SLAs or universal accuracy percentages:

| Area | Current measurement | Interpretation |
|---|---|---|
| Required production retrieval | 190/192 compiled; 172/192 delivered | Early terminal Pyrrho verdicts can stop before later ranked evidence |
| Query-shape suite | 60/60 | Measured deterministic shape coverage |
| Intentional limitation suite | 51/52 compiled; 48/52 delivered; 31/60 complete | Retrieval, delivery, and Pyrrho boundaries are reported separately |
| Local source indexing | 60.8 files/s core; 51.6 files/s mixed | Small local fixture corpora, source-only |
| NapierOne scale indexing | 4,994/5,005 at 7.27 files/s | Eleven malformed/ultra-wide CSV failures were explicit |
| Broad BEIR | 0.4239 delivered macro nDCG@10 | Biomedical, financial, and scientific tasks |
| Frozen semantic BEIR | 0.6519 delivered macro nDCG@10 | ArguAna/Quora task boundary |
| Enterprise holdout | 0.5780 delivered nDCG@10 | Multi-document ranking is the clearest quality weakness |
| Matched SciFact latency | 7.43s mean; 6.77s p50; 12.56s p95 | Local benchmark machine, warm matched sample |
| Enterprise warm probes | 13.092s and 19.889s | 511,961-file source index after closure fixes |

Thirteen broad-BEIR SciFact queries still lost their final judged-relevant
candidate during compilation; every one contained a structured scientific
identifier. Fitz-Sage will not add case-specific separator, abbreviation, or
naming heuristics to repair those user-owned equivalences.

See [Benchmarks](BENCHMARK.md) for methodology, component ablations, confidence
intervals, ingestion recovery evidence, and exact run revisions. See the
[EnterpriseRAG-Bench report](evaluation/enterprise-rag-bench-2026-08-01.md) and
[semantic holdout report](evaluation/beir-semantic-holdout-2026-07-30.md) for
the frozen external evaluations.

Use this file as the public product boundary. Remove a limitation only when a
general implementation improvement and a relevant benchmark demonstrate that
the boundary changed.
