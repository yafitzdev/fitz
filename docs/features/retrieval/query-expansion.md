# Query Expansion (Synonym/Acronym Variations)

## Problem

Users often use different terminology than what appears in documents:

- "How do I fetch employee data?" — document uses "retrieve" or "get"
- "How does the db connection work?" — document uses "database"
- "What failures can occur?" — document uses "errors" or "exceptions"

BM25 token matching catches *some* of this (shared stems, casing) but
not enough — it doesn't know `fetch` ↔ `retrieve` or `db` ↔ `database`.
The bridge lives on the query side: synonym and acronym terms are added to the
query's keyword set, which retrieval searches as one extra BM25 leg.

## Solution: synonym/acronym term expansion

`expand_terms()` adds dictionary synonym/acronym terms to the query's
keyword set:

```
Original query:   "How do I fetch the db config?"
                              ↓
expand_terms() →   ["get", "retrieve", "database", "datastore",
                    "configuration"]
                              ↓
Merged into the query-prep keyword set (alongside managed-Qwen
semantic keywords)
                              ↓
The keyword set runs as one extra FTS5 + bm25() retrieval leg
```

## How it works

### At query time

1. `expand_terms()` scans the query for known synonyms and acronyms
   (rule-based, no LLM).
2. The matched terms are merged into the query plan's keyword set alongside
   managed-Qwen semantic keywords.
3. The router runs that keyword set as one extra BM25 retrieval leg,
   pooled and ranked with the other strategy results.

### Expansion Rules

**Synonym Substitution:**
- `delete` ↔ `remove`, `erase`
- `create` ↔ `add`, `make`, `generate`
- `get` ↔ `retrieve`, `fetch`, `obtain`
- `update` ↔ `modify`, `change`, `edit`
- `error` ↔ `failure`, `exception`, `issue`
- And 40+ more common technical terms

**Acronym Expansion:**
- `api` → `application programming interface`
- `db` → `database`
- `auth` → `authentication`
- `config` → `configuration`
- `ml` → `machine learning`
- `rag` → `retrieval augmented generation`
- And 50+ more common acronyms

## Key Design Decisions

1. **Always-on** - Fused into the query-prep keyword set. No configuration.

2. **Rule-based** - No LLM calls. Fast and deterministic.

3. **Bidirectional synonyms** - Both directions work (fetch→retrieve, retrieve→fetch).

4. **Recall-stage** - Loose terms are fine — the cross-encoder reranker filters precision downstream.

## Files

- **`expand_terms()`:** `fitz_sage/retrieval/detection/detectors/expansion.py`
- **Keyword fusion:** `fitz_sage/engines/fitz_krag/query_batcher.py` (`_distribute` merges dict terms into `BatchResult.keywords`)
- **Retrieval leg:** `fitz_sage/engines/fitz_krag/retrieval/router.py`

Note: Query expansion uses dictionary-based matching (not LLM) for fast, deterministic results. Synonyms and acronyms are defined in the `SYNONYMS` and `ACRONYMS` dicts in `expansion.py`.

## Benefits

| Without Expansion | With Expansion |
|-------------------|----------------|
| "fetch" misses "retrieve" docs | "fetch" finds "retrieve" docs |
| "db" misses "database" docs | "db" finds "database" docs |
| User must guess exact terms | Natural language works |
| Lower recall | Higher recall |

## Example

**Query:** "How does the db connection work?"

**`expand_terms()` adds:** `database`, `datastore` (from `db`)

**Result:** Those terms join the keyword set; the keyword leg's BM25
search surfaces documents that say "database connection" even though
the user typed "db".

## Performance

- `expand_terms()` is microsecond-fast (rule-based dictionary lookup).
- It adds no endpoint call — dictionary terms ride alongside the managed-Qwen
  semantic keyword expansion that already runs in the default path.
- The keyword set adds one extra BM25 leg; FTS5 + `bm25()` is sub-10 ms
  per call on typical collections.

## Dependencies

- No LLM required (dictionary-based expansion).
- Fast, deterministic synonym/acronym matching.
- To add new synonyms or acronyms, edit the dicts in
  `detection/detectors/expansion.py`.

## Related

- [Sparse Search (FTS5 + bm25)](sparse-search.md) — what each expanded
  query actually hits
- [Keyword Vocabulary](keyword-vocabulary.md) — exact-match identifiers
  (complements synonym expansion)
- [Multi-Query RAG](multi-query-rag.md) — for long compound queries
  the rule-based expander can't handle
