# Query Expansion (Synonym/Acronym Variations)

## Problem

Users often use different terminology than what appears in documents:

- "How do I fetch employee data?" — document uses "retrieve" or "get"
- "How does the db connection work?" — document uses "database"
- "What failures can occur?" — document uses "errors" or "exceptions"

BM25 token matching catches *some* of this (shared stems, casing) but
not enough — it doesn't know `fetch` ↔ `retrieve` or `db` ↔ `database`.
With embeddings removed in v0.12.0, the bridging job falls on the
query side instead of the index side: expand the query into multiple
phrasings, run each, fuse.

## Solution: lightweight query expansion

Expand queries with synonym and acronym variations before BM25:

```
Original query:     "How do I fetch employee data?"
                              ↓
Expanded queries:   ["How do I fetch employee data?",
                     "How do I retrieve employee data?",
                     "How do I get employee data?"]
                              ↓
                    Each variation runs FTS5 + bm25()
                              ↓
                    Results merged via Reciprocal Rank Fusion
```

## How it works

### At query time

1. Query is analysed for known synonyms and acronyms (rule-based, fast).
2. Up to four additional query variations are generated.
3. Each variation hits the BM25 index.
4. Results are merged with Reciprocal Rank Fusion (`1 / (60 + rank)`).

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

1. **Always-on** - Baked into the KRAG retrieval router. No configuration.

2. **Rule-based** - No LLM calls. Fast and predictable.

3. **Bidirectional synonyms** - Both directions work (fetch→retrieve, retrieve→fetch).

4. **Case-preserving** - Preserves first character case of replaced word.

5. **Limit expansions** - Maximum 4 additional variations to control latency.

6. **RRF fusion** - Same RRF fusion used across multi-query expansion.

## Files

- **Expansion detector:** `fitz_sage/retrieval/detection/detectors/expansion.py`
- **Integration:** `fitz_sage/engines/fitz_krag/retrieval/router.py`

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

**Expanded to:**
1. "How does the db connection work?" (original)
2. "How does the database connection work?" (acronym expansion)
3. "How does the datastore connection work?" (synonym)

**Result:** Documents mentioning "database connection" are now found even though the user said "db".

## Performance

- Expansion is fast (microseconds, rule-based).
- One BM25 search per variation; FTS5 + `bm25()` is sub-10 ms per
  call on typical collections.
- RRF merge is in-memory.

Typical overhead: 2–4× search time for 3–5 variations. Negligible
relative to the LLM synthesizer step that follows.

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
