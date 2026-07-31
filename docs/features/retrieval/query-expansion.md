# Semantic Query Keywords

## Purpose

BM25 is fitz-sage's central recall mechanism, so lexical mismatch matters. A
query may say `fetch` while a relevant source says `retrieve`, or use `db` while
the corpus says `database`.

The default query path asks the managed local Qwen model for a small set of
semantic keywords. The router keeps the original query leg and runs the merged
deterministic and semantic keywords as one additional BM25 leg.

```text
original query --------------------------> BM25 leg --\
deterministic terms + Qwen suggestions --> BM25 leg ----+-> fused candidate budget
other query-shape variations ------------> BM25 legs --/
```

The merged candidates still pass through the ONNX cross-encoder reranker and
Pyrrho governance decision.

## Query-Time Flow

1. The deterministic planner extracts terms exactly as written by the user.
2. Managed Qwen proposes related retrieval keywords and short phrases.
3. fitz-sage de-duplicates the two sets without rewriting the source data.
4. The router searches the original query and the merged keyword set as
   separate BM25 legs.
5. Results from all recall legs are de-duplicated and ranked into one bounded
   candidate pool.

This path uses the managed ONNX Qwen runtime. It does not require a configured
endpoint model.

## Boundary

Semantic keywords are recall suggestions, not equivalence declarations.
fitz-sage does not include a fixed synonym/acronym dictionary and does not
silently canonicalize identifiers.

In particular:

- `AX-156`, `AX_156`, `AX 156`, and `AX156` remain distinct strings.
- Private abbreviations are not guaranteed to expand unless the corpus defines
  them or the user preprocesses the data.
- A Qwen suggestion can surface a candidate, but the suggestion does not prove
  that two terms mean the same thing in the user's domain.
- Literal retrieval still runs when a semantic suggestion is poor, but recall
  legs share a fixed output budget. Extra candidates can change ordering or
  displace a useful literal candidate.

Users who require guaranteed domain mappings should normalize their data and
queries outside fitz-sage. A public mapping-table hook is not currently part of
the package API.

## Measured Behavior

The frozen 2026-07-30 ArguAna/Quora holdout found that the current managed
Qwen path did not improve low-overlap recall consistently. Without reranking,
it reduced two-dataset macro recall nDCG@10 by `0.0106`, reduced final nDCG@10
by `0.0072`, and added `2.44s` per query. The Quora regressions were conclusive
under paired 95% bootstrap intervals.

With reranking active, Qwen changed macro final nDCG@10 by only `+0.0022`
while adding `2.06s`; the per-dataset effects were inconclusive. This does not
rule out better expansion models or fusion policies. It establishes only that
the current model did not earn its cost on these two BEIR tasks.

This result does not remove the architectural need for semantic expansion.
BM25 cannot match meaning that shares no useful lexical tokens, and the
holdout is a lexical-overlap proxy rather than an application-shaped
company-document evaluation. The default Qwen path remains the package's
best-effort general-language bridge while that broader evidence is gathered.

See
[BEIR Semantic Holdout](../../evaluation/beir-semantic-holdout-2026-07-30.md)
for the selection method, complete scores, and boundaries.

## Implementation

- Query keyword prompt and parsing:
  `fitz_sage/engines/fitz_krag/query_batcher.py`
- Query-time keyword merge:
  `fitz_sage/engines/fitz_krag/query_pipeline.py`
- BM25 keyword leg:
  `fitz_sage/engines/fitz_krag/retrieval/router.py`

## Related

- [Sparse Search](sparse-search.md)
- [Keyword Vocabulary](keyword-vocabulary.md)
- [Managed Models](../../MANAGED_MODELS.md)
