<!-- docs/features/retrieval/keyword-vocabulary.md -->
# Exact Identifier Matching

## Problem

Identifier lookups need exact matching, not soft similarity:

- **Q:** "What happened with `TC_1000`?"
- **Soft similarity:** treats `TC_1000` ≈ `TC_2000` — they look alike,
  so the wrong test case can come back.
- **Expected:** only `TC_1000`.

Test case IDs, ticket numbers, version strings, error codes, function
names, and similar tokens need **exact lexical matching**, not vector
similarity.

## Solution: SQLite FTS5 + native bm25()

fitz-sage runs no embedding model. All text search is lexical: typed
units (code symbols, document sections) are indexed in SQLite FTS5, and
ranking uses SQLite's native `bm25()` function. Because FTS5 matches on
literal tokens, an identifier like `TC_1000` or `E_AUTH_401` matches
the unit that actually contains that token — not a vaguely similar one.

```
Q: "error code E_AUTH_401"
     │
     ▼
FTS5 query over krag_section_fts / krag_symbol_fts
     │
     ▼
bm25() ranks units containing the literal token E_AUTH_401
     │
     ▼
Result: the section that defines E_AUTH_401, not E_AUTH_403
```

This is the **sparse-search** feature — see
[Sparse Search (FTS5 + bm25)](sparse-search.md) for the full mechanics.
There is no separate vocabulary store and no identifier pre-filter
stage.

## How it works

### At ingestion

1. **Typed-unit extraction.** Code files are parsed into symbols,
   documents into sections. The raw text — including every identifier
   it contains — is stored verbatim.

2. **FTS5 indexing.** Symbol and section text is indexed into FTS5
   virtual tables (`krag_symbol_fts`, `krag_section_fts`). The FTS5
   tokenizer keeps identifier tokens intact.

3. **Keyword enrichment.** `KragEnricher` extracts exact-match
   identifiers as `keywords` on each unit (function names, class
   names, IDs, abbreviations). These keywords are stored on the unit
   and are themselves searchable text.

### At query time

1. **FTS5 match.** The query runs as an FTS5 MATCH over the symbol and
   section indexes.
2. **bm25() ranking.** SQLite's native `bm25()` scores the matches —
   units containing the literal identifier rank highest.
3. **No special-casing.** Identifier queries and ordinary keyword
   queries take the exact same path; there is no separate detector or
   pre-filter.

## Key design decisions

1. **No embeddings at all.** fitz-sage is lexical-only, so exact
   identifier matching is the default behavior, not an add-on.
2. **One search path.** Identifiers and ordinary terms share the FTS5 +
   `bm25()` path — no vocabulary table, no pre-filter, no fallback
   branch.
3. **SQLite-native.** The FTS5 index lives in the same per-collection
   `.db` file as everything else. Deleting the collection file deletes
   the index.

## Configuration

None for search. FTS5 + `bm25()` are always active. Keyword enrichment is part
of the required ingestion contract and uses the managed Qwen ONNX runtime.

## Files

| Component             | Path                                                        |
| --------------------- | ----------------------------------------------------------- |
| FTS5 / bm25 search    | `fitz_sage/engines/fitz_krag/retrieval/` (section + symbol search strategies) |
| Symbol / section FTS5 indexes | `fitz_sage/engines/fitz_krag/ingestion/schema.py`   |
| Keyword enrichment    | `fitz_sage/engines/fitz_krag/ingestion/enricher.py` (`KragEnricher`) |

## Identifier types matched exactly

| Type            | Examples                                          |
| --------------- | ------------------------------------------------- |
| Test cases      | `TC-1001`, `testcase_42`, `TEST_AUTH`             |
| Tickets         | `JIRA-4521`, `BUG-789`, `ISSUE-123`               |
| Error codes     | `E_AUTH_401`, `ERR_TIMEOUT`                       |
| Versions        | `v2.0.1`, `1.0.0-beta`, `3.5`                     |
| Code classes    | `AuthService`, `UserController`, `PaymentHandler` |
| Code functions  | `handle_login()`, `process_payment()`             |
| Model numbers   | `X100`, `Model-Y200`, `SKU-4567`                  |

## Example

**Query:** "What tests failed in TC-1001?"

FTS5 + `bm25()` ranks units containing the literal token `TC-1001`
above units mentioning `TC-1002` or `TC-1003`:

1. `TC-1001 status: FAIL (assertion error)`
2. `TC-1001 detailed logs: line 42 failed`
3. `TC-1001 reproduction steps`

Because the match is lexical, the wrong test case is not retrieved
just because it looks similar.

## Dependencies

- SQLite + FTS5 (already required by the rest of fitz-sage).
- No embedding model, no vocabulary store.

## Related

- [Sparse Search (FTS5 + bm25)](sparse-search.md) — the FTS5 + native
  `bm25()` retrieval that delivers exact-identifier matching
- [Query Expansion](query-expansion.md) — handles synonyms and acronym
  expansion (the soft-matching counterpart)
- [Multi-Query RAG](multi-query-rag.md) — long queries may contain
  multiple identifiers to match
