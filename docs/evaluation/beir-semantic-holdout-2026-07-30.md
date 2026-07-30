# BEIR Semantic Holdout (2026-07-30)

This is the first frozen external measurement aimed specifically at managed
semantic query expansion. Low lexical overlap is used as a proxy for ordinary
paraphrase and vocabulary mismatch; this is not a synonym-only dataset and
does not test identifier normalization.

The result does not support paying for the current expansion model on every
query. The INT8 reranker remains useful overall, but its value and cost depend
strongly on query and document shape.

## Run Identity

- Run ID: `1785447318-d0a05a18`
- Git commit: `d855e4bec89b7f74959902e3859b5647b564230d`
- Worktree at run start: clean
- Frozen manifest:
  `beir-semantic-vocabulary-holdout-v1`
- Canonical manifest digest:
  `606baf0b2be00b78aecc737a9d2f550521254be5e2845232029a69b824b1df78`
- Queries: 120 ArguAna and 120 Quora queries
- Executions: 960 Fitz-Sage queries across four variants
- Expansion model: `onnx-community/Qwen3-0.6B-DQ-ONNX`
- Reranker: INT8 `Alibaba-NLP/gte-reranker-modernbert-base`
- Resumed queries: 0
- Runtime: 6,975 seconds, about 1 hour 56 minutes
- Child operational gates: all passed
- Paired measurement-integrity gate: passed with no failures

No relevance score was used to select a query. No retrieval behavior was
changed after inspecting scores from this holdout.

## Method

The fixture was committed before retrieval was run. For each dataset, all
eligible judged queries were ordered by their maximum case-folded token-set
Jaccard overlap with a judged-relevant document. They were split into low,
medium, and high lexical-overlap tertiles, then 40 queries per tertile were
selected by deterministic SHA-256 ordering with seed `20260730`.

The two datasets exercise different shapes:

- ArguAna uses long argumentative passages as queries and asks retrieval to
  find the best counterargument among 8,674 documents.
- Quora uses short natural-language questions and asks retrieval to find
  duplicate questions among 522,931 documents.

The same verified source-only indexes, query order, candidate budgets,
compiler, and exact Pyrrho runtime were used for every variant:

| Variant | Managed Qwen keywords | INT8 cross-encoder |
|---|---:|---:|
| `literal` | off | off |
| `expansion` | on | off |
| `reranker` | off | on |
| `full` | on | on |

The local plain-BM25 comparison searches whole projected documents. Fitz-Sage
variants retain typed retrieval, deterministic planning, evidence compilation,
and Pyrrho. Plain BM25 is therefore a useful quality reference, not a
latency-equivalent product path.

## Quality

| Dataset | Variant | Recall nDCG@10 | Final nDCG@10 | Delivered nDCG@10 | Recall@50 | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| ArguAna | plain BM25 | 0.4652 | - | - | 0.9250 | - |
| ArguAna | `literal` | 0.4509 | 0.4413 | 0.4260 | 0.9000 | 6.21s |
| ArguAna | `expansion` | 0.4463 | 0.4439 | 0.4311 | 0.9000 | 8.97s |
| ArguAna | `reranker` | 0.4509 | 0.4562 | 0.4433 | 0.9000 | 11.16s |
| ArguAna | `full` | 0.4463 | 0.4579 | 0.4445 | 0.9000 | 13.00s |
| Quora | plain BM25 | 0.7244 | - | - | 0.9245 | - |
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

## Component Effects

### Managed Qwen expansion

Without reranking, Qwen changed the two-dataset macro scores by:

- recall nDCG@10: `-0.0106`
- final nDCG@10: `-0.0072`
- delivered nDCG@10: `-0.0060`
- mean latency: `+2.44s`

On Quora, the recall change was `-0.0165` with a paired 95% interval of
`[-0.0331, -0.0036]`. The final change was `-0.0170` with an interval of
`[-0.0325, -0.0039]`. This is measured harm, not statistical noise under the
chosen paired analysis.

On ArguAna, the recall change was `-0.0047` and the final change was `+0.0026`;
both intervals crossed zero. There was no consistent gain in the low-overlap
stratum. The only conclusive low-overlap effect was an ArguAna final-ranking
gain when reranking was also enabled, but it did not repeat on Quora.

With reranking already active, adding Qwen changed macro final nDCG@10 by
`+0.0022` and delivered nDCG@10 by `+0.0020`, while adding `2.06s`. Those small
changes were inconclusive on both datasets.

Per-query comparison without reranking found the following nDCG@10 changes.
A hit means that the stage's returned ranking contained any judged-relevant
document: up to 50 documents at recall and up to 10 at final selection.

| Dataset | nDCG wins | nDCG losses | Ties | Relevant hits gained | Relevant hits lost |
|---|---:|---:|---:|---:|---:|
| ArguAna recall | 10 | 15 | 95 | 1 | 1 |
| ArguAna final | 10 | 13 | 97 | 1 | 0 |
| Quora recall | 8 | 17 | 95 | 0 | 1 |
| Quora final | 8 | 17 | 95 | 0 | 3 |

A post-run output diagnostic helps explain the weak result:

- ArguAna produced 951 phrases. Only 6.5% contained a new token that occurred
  in a judged-relevant document; 36/120 queries had any such bridge token.
- Quora produced 890 phrases. Only 6.3% contained a new token that occurred in
  a judged-relevant document; 22/120 queries had any such bridge token.
- Some outputs echoed the query or added generic terms such as `search`,
  `search query`, and `search keywords`.

The original query is still searched, but semantic terms run as an additional
BM25 leg and all legs compete within a fixed candidate budget. Weak expansion
can therefore change ranking or displace a useful literal tail candidate. The
trace analysis supports this as a plausible mechanism; it is not a separate
causal experiment.

The current managed Qwen path is not a reliable semantic bridge and its
always-on cost is not justified by this holdout. This conclusion applies to
the current model, prompt, and fusion behavior. It does not show that all
semantic query expansion is useless.

### INT8 reranker

Without Qwen, reranking changed final nDCG@10 by:

| Dataset | Mean delta | Paired 95% interval | Mean latency delta |
|---|---:|---:|---:|
| ArguAna | +0.0149 | [-0.0504, +0.0799] | +4.95s |
| Quora | +0.0518 | [+0.0075, +0.1003] | +0.45s |

The Quora gain is clear and inexpensive. The ArguAna result is positive on
average but inconclusive and much slower because the inputs are long
arguments. Across both datasets, reranking improved macro final nDCG@10 by
`0.0333` and delivered nDCG@10 by `0.0345` for `2.70s`.

The reranker should remain part of the pipeline. This result argues for
continued profile-aware budgeting and measurement by query shape, not one
global latency or quality claim.

### Literal retrieval

Literal Fitz-Sage substantially exceeded whole-document BM25 on Quora:
`0.8049` versus `0.7244` final nDCG@10. It trailed BM25 on ArguAna:
`0.4413` versus `0.4652`.

The typed lexical architecture is strong for short question-like retrieval.
Long passage-as-query argument matching remains a weaker shape. That is a
measured retrieval boundary, not a data-cleanup problem.

## Pyrrho Boundary

Compiled and delivered rankings were identical, so Fitz-Sage returned the
exact Pyrrho evidence decision. On ArguAna, long argumentative inputs often
triggered comparison or temporal query-shape signals. Two full-pipeline
queries had a judged-relevant final candidate but no judged-relevant compiled
evidence.

Current Pyrrho behavior is accepted for this Fitz-Sage release. No local
governance guard was added. Absolute ArguAna delivered scores include that
accepted behavior, while the paired Qwen and reranker comparisons remain
aligned because every variant used the same Pyrrho runtime.

## Operational Findings

Two failures were fixed before any holdout scores were inspected:

1. Long repeated Qwen inference could terminate in ONNX Runtime GenAI with
   `bad allocation`. Generation is now bounded to 128 output tokens and an
   8,192-token runtime context, request-local generators are released
   explicitly, and that native allocation failure receives one retry.
2. Qwen could emit one unterminated comma-separated string instead of separate
   JSON array items. The prompt now states the item contract explicitly and
   output is capped at ten phrases. All 120 ArguAna prompts replayed
   successfully after the fix.

These are general runtime and output-contract fixes. They do not inspect
retrieval scores or encode benchmark terms.

The four variants reused persisted source indexes only after checking the
source path, expected document count, failure state, source-ID mapping, and
every persisted content hash against the deterministic adapter mapping.
Quora's 522,931-document index validated in about 5.5 seconds. This strict
benchmark reuse avoids conflating unchanged source traversal with
query-component timing.

Ordinary no-change `point()` still walks and hashes source files. On the same
522,931-file projection it did not complete within a 20-minute diagnostic
window. That is a real extreme-file-count startup limitation; benchmark reuse
does not change product behavior.

## Decision And Next Measurement

- Keep this holdout frozen and evaluation-only.
- Do not tune term filters or fusion weights against these 240 scored queries.
- Build a separate development set before changing expansion or fusion.
- Re-run this holdout once after the planned managed-model replacement.
- Keep identifier aliases, private abbreviations, and source cleanup outside
  Fitz-Sage.
- Keep governance changes in Pyrrho.

The generated query-level JSON and Markdown reports remain under
`benchmarks/results/`, which is intentionally ignored. This document is the
committed result record.
