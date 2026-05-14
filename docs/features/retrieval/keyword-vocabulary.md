# Keyword Vocabulary (Exact Match)

## Problem

BM25 token matching is forgiving — `TC-1001` and `TC-1002` share the
prefix `TC` and are tokenised similarly. For most queries that's
fine, but for identifier lookups it's exactly wrong:

- **Q:** "What happened with `TC-1001`?"
- **Pure BM25:** also returns `TC-1002`, `TC-1003`, `TC-999` —
  the FTS5 tokeniser treats them as similar terms.
- **Expected:** only `TC-1001`.

Test case IDs, ticket numbers, version strings, function names, and
similar tokens need **exact matching with controlled variation**, not
soft token overlap.

## Solution: keyword vocabulary pre-filter

fitz-sage auto-detects identifiers during ingestion, stores them in a
per-collection vocabulary table, and uses that vocabulary to
pre-filter candidates before BM25.

```
Q: "What happened with TC-1001?"
     │
     ▼
Vocabulary lookup → matches { TC-1001 }
     │
     ▼
BM25 search restricted to addresses
that mention TC-1001 (or a variation)
     │
     ▼
Result: only TC-1001 content, never TC-1002
```

## How it works

### At ingestion

1. **Pattern detection.** Regex patterns surface candidate identifiers:
   - Test cases: `TC-\d+`, `testcase_\d+`, `TEST_\w+`
   - Tickets: `JIRA-\d+`, `BUG-\d+`, `ISSUE-\d+`
   - Versions: `v?\d+\.\d+\.\d+`, `\d+\.\d+-beta`
   - Code identifiers: `[A-Z][a-zA-Z]+Service`, `\w+Controller`

2. **Vocabulary storage (SQLite).**

   ```sql
   -- per-collection .db
   CREATE TABLE IF NOT EXISTS keywords (
       id            TEXT PRIMARY KEY,
       category      TEXT NOT NULL,
       match         TEXT NOT NULL DEFAULT '[]',  -- JSON array of variations
       occurrences   INTEGER NOT NULL DEFAULT 1,
       first_seen    TEXT,
       user_defined  INTEGER NOT NULL DEFAULT 0,
       auto_generated TEXT NOT NULL DEFAULT '[]'
   );
   ```

3. **Variation normalisation.** A single keyword gets a JSON array
   of normalised variations (`TC-1001` ↔ `tc-1001` ↔ `TC_1001` ↔
   `TC 1001`) so the runtime can match across surface differences.

### At query time

1. **Detect identifiers in the question.** Same regex set runs over
   the query.
2. **Pre-filter.** If any identifier matches the vocabulary, restrict
   the BM25 candidate pool to addresses that reference at least one
   matching variation.
3. **Soft fallback.** If no identifier is detected (or no matches in
   the vocabulary), the pipeline runs unfiltered BM25 — no behaviour
   change for ordinary questions.
4. **Case- and delimiter-insensitive matching.**

## Key design decisions

1. **Always-on.** Baked into ingestion and retrieval. No configuration.
2. **Pre-filter, not post-filter.** Filtering happens before BM25 so
   the candidate pool stays small.
3. **Graceful degradation.** No identifier → unfiltered pipeline.
4. **SQLite-native.** Vocabulary lives in the same `.db` file as the
   rest of the collection. Deleted automatically when the collection
   file is removed.

## Configuration

None. Internal knobs in `KeywordExtractor`:

- Pattern regex set
- Minimum keyword length (default 3)
- Maximum keywords per chunk (default 50)

## Files

| Component             | Path                                                        |
| --------------------- | ----------------------------------------------------------- |
| Vocabulary module     | `fitz_sage/retrieval/vocabulary/`                           |
| SQLite store          | `fitz_sage/retrieval/vocabulary/store.py` (`VocabularyStore`) |
| Query-time filter     | KRAG retrieval router (`fitz_sage/engines/fitz_krag/retrieval/router.py`) |
| Ingestion hook        | `fitz_sage/ingestion/enrichment/modules/chunk/keywords.py`  |

## Detected identifier types

| Type            | Examples                                          |
| --------------- | ------------------------------------------------- |
| Test cases      | `TC-1001`, `testcase_42`, `TEST_AUTH`             |
| Tickets         | `JIRA-4521`, `BUG-789`, `ISSUE-123`               |
| Versions        | `v2.0.1`, `1.0.0-beta`, `3.5`                     |
| Code classes    | `AuthService`, `UserController`, `PaymentHandler` |
| Code functions  | `handle_login()`, `process_payment()`             |
| Model numbers   | `X100`, `Model-Y200`, `SKU-4567`                  |

## Example

**Query:** "What tests failed in TC-1001?"

**Pure BM25:**
1. `TC-1002 test results: PASS`
2. `TC-1003 failure log: timeout error`
3. `TC-1001 status: FAIL (assertion error)`

**With keyword vocabulary pre-filter:**
1. `TC-1001 status: FAIL (assertion error)`
2. `TC-1001 detailed logs: line 42 failed`
3. `TC-1001 reproduction steps`

Only addresses containing `TC-1001` (or a recognised variation) make
it to the BM25 stage.

## Dependencies

- SQLite + FTS5 (already required by the rest of fitz-sage).
- `VocabularyStore` lives in the per-collection `.db`; deleting the
  collection file deletes the vocabulary too.

## Related

- [Sparse Search (FTS5 + bm25)](sparse-search.md) — handles soft
  token matching once the vocabulary has restricted the candidate pool
- [Query Expansion](query-expansion.md) — handles synonyms; vocabulary
  handles exact identifiers
- [Multi-Query RAG](multi-query-rag.md) — long queries may contain
  multiple keywords to filter on
