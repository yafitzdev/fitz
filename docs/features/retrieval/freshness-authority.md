# Freshness Boosting

## Problem

Standard RAG treats all documents equally. This fails when:

- "What's the latest status on feature X?" — Can't distinguish old vs new docs

Recency-sensitive questions need the retrieval ranking to favor
recently-modified content.

## Solution: Intent-Triggered Recency Boosting

Detect recency intent from the query and boost recently-modified files
accordingly:

- **Recency keywords** ("latest", "recent", "current", "new",
  "updated", "newest") → boost newer files

## How It Works

Detection is **LLM-based**. The `FreshnessModule` contributes a prompt
fragment to the single batched query-intelligence call made by
`QueryBatcher`; the LLM sets a `boost_recency` flag from the query
intent.

```
Query comes in
    │
    ├─ LLM detects recency intent? (asks about "latest", "recent", "current")
    │       │
    │       ▼
    │   Multiplicative score boost for recently-modified files
    │
    └─ No recency intent? → Pass through unchanged
```

## Key Design Decisions

1. **On by default, intent-triggered** — part of every query pass. Only
   activates when the query signals recency intent.

2. **Metadata captured at ingestion** — file modification timestamps
   are recorded during ingestion.

3. **Multiplicative score boost** — recently-modified files get their
   relevance score scaled up; it does not override relevance ordering
   for non-recency queries.

4. **Graceful degradation** — if timestamp metadata is missing, units
   pass through unchanged.

## Example

**Query:** "What's the latest status on the battery warranty change?"

**Before recency boost:**
1. notes/old_review_2023.md (score: 0.85)
2. status/warranty_update_2026.md (score: 0.82)

**After recency boost:** (recency keyword "latest" detected)
1. status/warranty_update_2026.md (score: boosted — recently modified)
2. notes/old_review_2023.md (score: 0.85)

## Configuration

No configuration required. The feature is built into the retrieval
pipeline.

## Intent Keywords

**Recency triggers:**
- "latest", "recent", "current", "new", "updated", "newest"

## Implementation

- **Detection module:** `fitz_sage/retrieval/detection/modules/freshness.py`
- **Query intelligence:** `fitz_sage/engines/fitz_krag/query_batcher.py` (`QueryBatcher`)

Detection is LLM-based. The `FreshnessModule` contributes a prompt
fragment to the batched `QueryBatcher` call and parses a `boost_recency`
flag from the combined response.

## Benefits

| Without Freshness | With Freshness |
|-------------------|----------------|
| Old docs rank equally | Recent docs boosted when asked |
| User filters manually | Intent-based automatic boosting |

## Related Features

- [**Temporal Queries**](temporal-queries.md) - Time-based retrieval (freshness complements time filtering)
- [**Aggregation Queries**](aggregation-queries.md) - List/count detection (another detection module)
- [**Comparison Queries**](comparison-queries.md) - Entity comparison (another detection module)
