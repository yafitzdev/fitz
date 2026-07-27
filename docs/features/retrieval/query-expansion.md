# Semantic Query Keywords

## Purpose

BM25 is fitz-sage's central recall mechanism, so lexical mismatch matters. A
query may say `fetch` while a relevant source says `retrieve`, or use `db` while
the corpus says `database`.

The default query path asks the managed local Qwen model for a small set of
semantic keywords. Those suggestions are merged with the literal query terms
and searched as an additional BM25 leg.

```text
literal query terms ─┐
                    ├─> merged keyword set ─> FTS5 + bm25()
Qwen suggestions ───┘
```

The merged candidates still pass through the ONNX cross-encoder reranker and
Pyrrho governance decision.

## Query-Time Flow

1. The deterministic planner extracts terms exactly as written by the user.
2. Managed Qwen proposes related retrieval keywords and short phrases.
3. fitz-sage de-duplicates the two sets without rewriting the source data.
4. The router searches the merged keyword set as one broad-recall BM25 leg.

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
- Deterministic retrieval continues to work from literal query terms if a
  semantic suggestion is not useful.

Users who require guaranteed domain mappings should normalize their data and
queries outside fitz-sage. A public mapping-table hook is not currently part of
the package API.

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
