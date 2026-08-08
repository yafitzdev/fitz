# Sparse Search With SQLite FTS5

Sparse BM25 search is Fitz-Sage's central recall mechanism. It searches typed
source indexes and feeds one fused candidate pool to the ONNX reranker.

## Indexed Surfaces

| Surface | FTS5 index | Indexed text |
|---|---|---|
| Document sections | `krag_section_fts` | section title and original content |
| Code symbols | `krag_symbol_fts` | original/split names, qualified name, signature, docstring/comment |
| Native table rows | `_table_row_fts` | serialized concrete row values |

Table metadata uses ordinary name/column lookup. Import edges and entity links
use normal SQLite indexes. There is no separate vocabulary FTS table.

## Query Construction

Application queries are not exposed as raw FTS5 syntax. `build_fts_query()`
extracts word tokens, quotes each token, and OR-joins them:

```text
error code E_AUTH_401
    -> "error" OR "code" OR "E_AUTH_401"
```

This keeps free-form punctuation and FTS operators from changing the query
grammar. Empty token sets return no sparse results.

SQLite FTS5 tokenization can make separator variants share lexical pieces, but
that is not an equivalence guarantee. Exact identifier anchoring and evidence
compilation still preserve the package's literal-input contract.

## Ranking Convention

FTS5 `bm25()` returns lower values for stronger matches. Store methods negate
the raw value before creating addresses so higher downstream scores mean better
matches. Strategy results are then fused and deduplicated before reranking.

The section store first ranks lightweight row identifiers and materializes only
the winning rows. This avoids reading every large section body merely to rank a
bounded result set.

## Role In The Pipeline

```text
literal query + semantic query terms + query-shape variations
    -> section / symbol / row sparse searches
    -> cross-strategy fusion
    -> bounded ONNX cross-encoder
    -> source reading and evidence compilation
```

Sparse recall intentionally favors breadth. The cross-encoder and compiler own
later precision; BM25 does not decide whether evidence is sufficient.

## Boundaries

- Lexically unrelated wording needs Qwen query terms, explicit corpus evidence,
  or user preprocessing.
- Private abbreviations and identifier aliases are not inferred.
- Very common terms can produce a noisy pool within finite candidate limits.
- FTS5 is not a dense semantic index and is not exposed as an end-user query
  language.

Measured relevance and timing are in [Benchmarks](../../BENCHMARK.md).

## Implementation

- `fitz_sage/engines/fitz_krag/ingestion/store_utils.py`
- `fitz_sage/engines/fitz_krag/ingestion/section_store.py`
- `fitz_sage/engines/fitz_krag/ingestion/symbol_store.py`
- `fitz_sage/tabular/store/sqlite.py`

## Related

- [Semantic Query Keywords](query-expansion.md)
- [Literal Identifier Search](keyword-vocabulary.md)
- [Reranking](reranking.md)
- [Unified Storage](../platform/unified-storage.md)
