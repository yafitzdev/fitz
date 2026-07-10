# Sparse Search (FTS5 + `bm25()`)

## Problem

Similarity-only retrieval can fail on exact terms:

- **Q:** "Find documents mentioning X100"
- **Similarity-only search:** returns Y200 docs (semantically similar
  model numbers)
- **Expected:** exact match on `X100`

Product codes, error messages, and technical identifiers need lexical matching.
fitz-sage handles that with a sparse retrieval stack wrapped in typed-unit
routing, structural expansion, and ONNX cross-encoder reranking.

## Solution: SQLite FTS5 with native `bm25()`

Every store (sections, symbols, table-store, vocabulary, entity graph)
maintains an **FTS5 external-content index** over its searchable
columns. Ranking uses SQLite's built-in `bm25()` function.

```
Query: "X100 battery specs"
         │
         ▼
   FTS5 MATCH on sections_fts / symbols_fts / etc.
         │
         ▼
   bm25(...) scores rows  (negative = better)
         │
         ▼
   sign-flipped to positive (higher = better)
         │
         ▼
   Top-K rows joined back to the base table → addresses
```

## How it works

### External-content tables

An FTS5 external-content table mirrors the searchable columns of a
base table without storing them twice on disk. Updates flow via
triggers; the index stays current automatically.

```sql
CREATE TABLE IF NOT EXISTS sections (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    title         TEXT,
    content       TEXT NOT NULL,
    summary       TEXT,
    ...
);

CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts
USING fts5(title, content, summary, content='sections', content_rowid='rowid');
```

### Native ranking

FTS5 ships `bm25(<fts_table>)` as a ranking function. It returns
negative numbers — lower = better. Production code flips the sign so
downstream consumers can treat **higher = better** (and the test suite
asserts this convention — see
[`test_section_store::test_returns_results_with_bm25_score`]).

```sql
SELECT s.*, bm25(sections_fts) AS rank
FROM sections_fts
JOIN sections s ON s.rowid = sections_fts.rowid
WHERE sections_fts MATCH ?
ORDER BY rank
LIMIT ?;
```

### Query syntax

FTS5 supports several query shapes:

| Form              | Example                              | Meaning                                |
| ----------------- | ------------------------------------ | -------------------------------------- |
| Bag of words      | `battery specifications`             | match either token, rank by BM25       |
| Phrase            | `"machine learning"`                 | adjacent tokens                        |
| Boolean           | `python AND django`                  | both required                          |
| NEAR              | `NEAR(token1 token2, 5)`             | within N tokens                        |
| Column-scoped     | `title:battery`                      | match only the `title` column          |
| Prefix            | `X100*`                              | tokens starting with `X100`            |

### Why FTS5 (and not a separate BM25 library)

- **No extra dependency.** FTS5 ships in stdlib `sqlite3`.
- **No separate index to sync.** External-content tables stay
  consistent with the base table via triggers.
- **Transactional with the rest of the row.** Insert a row, the FTS
  index updates in the same transaction.
- **Familiar.** SQL all the way down — no DSL, no operator overloading.

## Key design decisions

1. **Always on.** Every store creates its FTS5 sibling at schema-init
   time; no configuration knob.
2. **SQLite-native.** No external libraries (no rank_bm25, no whoosh).
3. **Sign-flipped score.** Production negates `bm25()` so the rest of
   the pipeline can use "higher = better" uniformly.
4. **Graceful fallback.** If the FTS5 query string can't be parsed,
   the store falls back to a `LIKE COLLATE NOCASE` scan.

## Configuration

None. The FTS5 indexes are built and queried automatically by the
KRAG retrieval strategies.

Internal parameters:

- `top_addresses` — how many BM25 candidates to fetch (default 50)
- `top_read` — how many to keep after expansion + rerank (default 50)

## Files

| Component              | Path                                                                  |
| ---------------------- | --------------------------------------------------------------------- |
| Section store FTS      | `fitz_sage/engines/fitz_krag/ingestion/section_store.py`              |
| Symbol store FTS       | `fitz_sage/engines/fitz_krag/ingestion/symbol_store.py`               |
| Table store FTS        | `fitz_sage/engines/fitz_krag/ingestion/table_store.py`                |
| Code search strategy   | `fitz_sage/engines/fitz_krag/retrieval/strategies/code_search.py`     |
| Section search strategy| `fitz_sage/engines/fitz_krag/retrieval/strategies/section_search.py`  |
| Connection manager     | `fitz_sage/storage/sqlite.py`                                         |

## Performance

- **Index overhead:** ~10–15% over the base content size.
- **Query latency:** sub-10 ms on collections in the hundreds-of-MB
  range; SQLite + FTS5 is fast.
- **No network hop.** Storage is a local file, so the lower bound
  on per-query retrieval is process-local I/O.

## Example

**Documents:**
- Doc A: "The X100 has a 5000 mAh battery with fast charging."
- Doc B: "The Y200 features similar battery capacity to other models."
- Doc C: "Battery specifications vary across the X-series lineup."

**Query:** `X100 battery`

**FTS5 + bm25 result:**
1. Doc A — both terms hit, prominent in title-equivalent
2. Doc C — partial (`X-series`) + `battery`
3. Doc B — `battery` only; no `X100`

Then the **ONNX cross-encoder reranker** re-orders this small
candidate set in a single forward pass, using `(query, doc)` joint
context that BM25 doesn't see.

## Related

- [Keyword Vocabulary](keyword-vocabulary.md) — exact-match identifier
  storage complements BM25
- [Semantic Query Keywords](query-expansion.md) — managed-Qwen keyword
  expansion feeding the BM25 query
- [Reranking](reranking.md) — ONNX cross-encoder reranker that runs after BM25
- [Unified Storage](../platform/unified-storage.md) — the SQLite +
  FTS5 layer the search runs on
