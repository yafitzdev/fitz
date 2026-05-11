# docs/roadmap/query-expansion-without-embeddings.md
# Query Expansion Without Embeddings — HyDE Replacement on BM25

## Problem

fitz-sage v0.12.0 dropped embeddings, vector DB, and HyDE. Retrieval
is now BM25 (FTS5 + `bm25()`) → LLMReranker → grounded synthesis. The
trade-off: BM25 has known vocabulary-mismatch weaknesses — *"physician"*
won't retrieve docs that say *"doctor"*, *"K8s"* won't retrieve
*"Kubernetes"*, and question-form queries don't lexically match
answer-form documents.

The LLMReranker compensates on the *top-N candidates BM25 already
surfaced*. If BM25 misses a relevant doc entirely (because the doc
uses different vocabulary than the query), no amount of reranking
recovers it.

This is the gap HyDE used to fill — it generated answer-shaped text
from the query, embedded that, and retrieved on semantic similarity.
The proposal here: **fill the gap without bringing embeddings back**.

## Approach 1 — Query-time LLM keyword expansion

At query time, call the chat model (fast tier) to expand the query
into alternate phrasings, synonyms, and likely document terms.
Fan-out each expansion through BM25, RRF-merge the candidate lists,
let the LLMReranker do its job.

This is the well-studied **Generative Query Expansion** / **Query2Doc**
technique (Wang et al., 2023; Mackie et al., 2023). Variants live in
LangChain (`MultiQueryRetriever`) and LlamaIndex (`HyDEQueryTransform`).

```
user query
  → LLM (fast tier): "List 5 alternate phrasings + synonyms + domain
                      terms that documents might use to answer this"
  → query + N expansions
  → FTS5 MATCH on each (parallel)
  → RRF-merge candidate sets
  → LLMReranker
  → answer
```

**Cost:** one extra LLM call per query (~0.5–2s on a local 7B model).
**Win:** recall recovery on vocabulary-mismatch queries.
**Loss:** per-query latency cost; LLM keywords may not match the
specific corpus vocabulary (e.g., LLM generates "physician" when the
docs say "MD" or "attending").

## Approach 2 — Vocabulary-grounded expansion via the enrichment bus

fitz-sage already builds a per-collection keyword vocabulary at
**ingestion time**. The `KragEnricher` extracts keywords + entities
for each chunk during ingest; the `VocabularyStore` consolidates
them per collection with variation tracking
(`fitz_sage/retrieval/vocabulary/store.py`). A `KeywordMatcher`
already exposes `find_in_query(q)` which returns the corpus keywords
mentioned in a user query, each with its full variation list
(`fitz_sage/retrieval/vocabulary/matcher.py`).

**The insight:** at query time we don't need to ask an LLM "what
synonyms might documents use?" — the documents have already told us,
during ingestion. The vocabulary is already there.

```
ingestion (one-time):
  chunk → KragEnricher → keywords + variations → VocabularyStore

query (zero LLM calls for expansion):
  user query
    → KeywordMatcher.find_in_query(q)   # vocab lookup, microseconds
    → matched Keywords with .match[]    # corpus-grounded variations
    → expand each into FTS5 OR clauses
    → FTS5 MATCH
    → LLMReranker
    → answer
```

**Cost:** zero query-time LLM calls for expansion. The LLM cost is
paid once at ingest (and was already being paid by the enricher).
**Win:** corpus-grounded — every expansion is a term that actually
appears in this collection. No drift, no hallucination.
**Loss:** bounded by the enricher's coverage. If the enricher missed
a synonym at ingest time, the query won't find it. New documents
need the vocabulary index to be updated (incremental merge is
already in `VocabularyStore.merge_and_save`).

## Approach 3 — Hybrid (recommended end state)

1. **Fast path:** `KeywordMatcher.find_in_query(q)` — if any vocab
   matches, expand via variations. Zero LLM calls.
2. **Fallback path:** if no vocab matches (rare query, novel domain
   term), call the chat LLM for a one-shot expansion. The latency
   hit only applies when vocab is genuinely insufficient.
3. **Always:** RRF-merge raw query results with expanded results so
   the BM25 baseline is preserved.

This collapses to Approach 2 on most queries and Approach 1 only when
the vocab gap is real.

## Existing infrastructure to leverage

- **`fitz_sage/retrieval/rewriter/rewriter.py`** — `QueryRewriter`
  currently does pronoun resolution, simplification, question →
  statement. Already in the engine init at
  `engine.py` `self._query_rewriter = QueryRewriter(...)`.
  Natural place to slot expansion (or its own `QueryExpander` step).
- **`fitz_sage/retrieval/vocabulary/store.py`** — `VocabularyStore`.
  Already loads per-collection. SQLite-backed since v0.12.0.
- **`fitz_sage/retrieval/vocabulary/matcher.py`** — `KeywordMatcher`
  with `find_in_query(q) → list[Keyword]`. Returns matched
  variations ready to be OR-joined into FTS5 syntax.
- **`enable_multi_query: bool = True`** in `FitzKragConfig` —
  multi-query fan-out already supported for long queries
  (`multi_query_min_length: int = 300`). The expansion path just
  becomes another source of fan-out queries.
- **`SectionStore.search_bm25()`** uses `_build_fts_query()` which
  OR-joins alphanumeric word tokens. Already accepts arbitrary FTS5
  query strings — a vocabulary-expanded query slots in unchanged.

So the implementation cost is small: maybe **100 lines + one prompt
file** for Approach 1, even less for Approach 2 (it's mostly wiring
existing `KeywordMatcher` into the retrieval flow).

## Phases

| Phase | Description | Effort | Status |
|-------|-------------|--------|--------|
| 1 | `QueryExpander` step that calls `KeywordMatcher.find_in_query` and OR-joins variations into FTS5 (Approach 2) | Low | Proposed |
| 2 | Benchmark vs. baseline BM25 on BEIR (gated on Task 12 — restore evaluation subpackage) | Low | Proposed |
| 3 | Add LLM-expansion fallback for vocab-miss queries (Approach 3 hybrid) | Medium | Proposed |
| 4 | If win, make default. If loss, document why and shelve. | — | Gated on Phase 2 |

## Open questions

1. **Does the existing `KragEnricher` produce keywords useful for
   retrieval, or are they too topical/narrow?** The vocabulary
   already has variation tracking — worth manually inspecting
   `VocabularyStore.load()` on the smoke corpus to see what's in it.
2. **How does corpus-grounded expansion compare to actual HyDE?**
   HyDE generated *answer-shaped* text — a paragraph that looks like
   an answer. Keyword expansion is shallower: alternate terms, not
   alternate phrasings of an answer. The reranker likely closes the
   gap, but worth measuring.
3. **Stale vocab on incremental ingest:** `VocabularyStore` already
   has `merge_and_save` for keyword preservation, but if the user
   adds a wholly new domain to a collection mid-life, the vocab may
   lag the corpus until reindex. How aggressively does it need to
   refresh?
4. **Cold-start corpora:** what happens on the very first query
   after `point()` when vocab isn't yet built? Today the
   `KeywordMatcher` returns nothing → query passes through unchanged
   to BM25. Acceptable, but worth confirming.

## Why this matters

If Approach 2 works, fitz-sage achieves **lexical retrieval with
semantic-coverage characteristics at zero query-time LLM cost** — a
genuine architectural win and a defensible "we don't need embeddings"
argument. The enrichment bus stops being just an ingest-time signal
and becomes the **retrieval intelligence layer** that fills the role
embeddings used to play.

If it doesn't work, we know empirically that embeddings (or HyDE,
or Approach 1's LLM-expansion) earn their keep — and we have the
BEIR measurement to defend that decision.

The bet is small, the upside is large, and the validation
infrastructure (BEIR) already needs rebuilding for Task 12 anyway.
Worth running.
