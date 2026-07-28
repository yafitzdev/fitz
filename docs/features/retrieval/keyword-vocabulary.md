<!-- docs/features/retrieval/keyword-vocabulary.md -->
# Literal Identifier Search

Identifiers such as test IDs, tickets, error codes, versions, and function names
are stored from source text without rewriting.

## Source Contract

Fitz-Sage does not silently declare separator variants equivalent:

- `TC-1001`
- `TC_1001`
- `TC 1001`
- `TC1001`

If those values mean the same thing in a user's domain, the user must normalize
the source/query or provide explicit alias evidence. See
[Limitations](../../../LIMITATIONS.md).

## Retrieval Path

```text
source text and typed units
    -> SQLite FTS5 index
query literal and semantic terms
    -> FTS5 MATCH + bm25()
    -> ONNX cross-encoder reranking
```

There is no ingestion-time keyword column, fixed synonym dictionary,
identifier-normalization pass, or alternate vocabulary store. Code retrieval
also searches extracted symbol names and qualified names.

SQLite tokenization can surface lexical neighbors. For example, punctuation may
split a value into component tokens. A retrieved neighbor is a candidate, not a
claim that two identifiers are equivalent. The reranker and Pyrrho see the raw
source form.

## Query-Time Expansion

The recall query may contain:

- literal terms from the original query;
- semantic keywords proposed by the managed local Qwen runtime;
- explicit mapping terms supplied by a future user-owned vocabulary hook.

Semantic terms are search suggestions. They do not rewrite stored data and do
not prove equivalence.

## Why No Automatic Cleanup

Identifier conventions are domain-specific. Treating punctuation variants as
aliases may improve one corpus and create false matches in another. Fitz-Sage
therefore keeps document cleanup and authoritative mappings outside the
universal package core.

## Files

| Component | Path |
|---|---|
| Section FTS5 schema | `fitz_sage/engines/fitz_krag/ingestion/schema.py` |
| Section BM25 | `fitz_sage/engines/fitz_krag/ingestion/section_store.py` |
| Symbol name/BM25 search | `fitz_sage/engines/fitz_krag/ingestion/symbol_store.py` |
| Query semantic keywords | `fitz_sage/engines/fitz_krag/query_batcher.py` |

## Related

- [Sparse Search](sparse-search.md)
- [Semantic Query Expansion](query-expansion.md)
- [Ingestion Pipeline](../../INGESTION.md)
