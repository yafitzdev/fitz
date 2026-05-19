# Temporal Queries (Time-Based Retrieval)

## Problem

Users often ask time-related questions that require special handling:

- "What happened in Q1 2024?" - Need to retrieve period-specific content
- "Compare version 1.0 and version 2.0" - Need content from both versions
- "What changed between 2023 and 2024?" - Need to compare time periods
- "What was the status before the merger?" - Need pre-event content

Standard semantic search doesn't understand temporal intent.

## Solution: Temporal Query Detection and Handling

Detect temporal intent and generate time-focused sub-queries:

```
Original query:     "What changed between Q1 and Q2 2024?"
                              ↓
                    Temporal Detection
                              ↓
Intent: CHANGE      Refs: [Q1, Q2 2024, 2024]
                              ↓
                    Generate Sub-Queries
                              ↓
Queries:            ["What changed between Q1 and Q2 2024?",
                     "What changed Q1",
                     "What changed Q2 2024"]
                              ↓
                    Search each → RRF merge
                              ↓
Result:             Chunks from both periods, ranked by relevance
```

## How It Works

### Temporal Intent Detection

The system detects five types of temporal intent:

| Intent | Triggers | Example |
|--------|----------|---------|
| COMPARISON | "between X and Y", "compare", "vs" | "Compare 2023 vs 2024" |
| TREND | "over time", "history", "evolution" | "How has revenue trended?" |
| POINT_IN_TIME | "as of", "in Q1", specific date | "Revenue in Q1 2024" |
| RANGE | "from X to Y", date ranges | "Between January and March" |
| SEQUENCE | "first/then/after", "what changed" | "What changed after Q1?" |

### Temporal Reference Extraction

Extracts time references from queries:

- **Quarters:** Q1, Q2, Q3, Q4 (with optional year)
- **Years:** 2023, 2024, etc.
- **Versions:** v1.0, version 2.0, etc.
- **Months:** January 2024, March, etc.
- **Relative:** last month, last year, yesterday, recently
- **Dates:** 01/15/2024, 2024-01-15

### Sub-Query Generation

For each temporal reference, generates focused sub-queries:

```
Query: "Compare version 1.0 and version 2.0"
  ↓
Sub-queries:
  1. "Compare version 1.0 and version 2.0" (original)
  2. "version 1.0" (focused on v1)
  3. "version 2.0" (focused on v2)
```

### Result Merging

Results from all sub-queries are merged using RRF:
- Chunks appearing in multiple time periods get higher scores
- Temporal references are tagged in chunk metadata

## Key Design Decisions

1. **Always-on** - Baked into the KRAG retrieval router. No configuration.

2. **Intent-first** - Detects intent before extracting references.

3. **Multi-query** - Generates sub-queries for each time period.

4. **RRF fusion** - Same RRF fusion used across multi-query expansion.

5. **Metadata tagging** - Chunks tagged with `temporal_refs` for downstream use.

## Files

- **Detection module:** `fitz_sage/retrieval/detection/modules/temporal.py`
- **Query intelligence:** `fitz_sage/engines/fitz_krag/query_batcher.py` (`QueryBatcher`)
- **Integration:** `fitz_sage/engines/fitz_krag/retrieval/router.py`

Detection is LLM-based. The `TemporalModule` contributes its prompt fragment to the single batched `QueryBatcher` call and parses `TemporalIntent` (COMPARISON, TREND, POINT_IN_TIME, RANGE, SEQUENCE) from the combined LLM response.

## Benefits

| Without Temporal Handling | With Temporal Handling |
|---------------------------|------------------------|
| "Q1 2024" treated as keywords | Q1 2024 context prioritized |
| Version comparisons miss one side | Both versions retrieved |
| "What changed" not understood | Change intent triggers multi-period search |
| Before/after ignored | Time constraints respected |

## Example

**Query:** "What changed between Q1 and Q2 2024?"

**Detection:**
- Intent: CHANGE
- References: [Q1, Q2 2024, 2024]

**Generated Queries:**
1. "What changed between Q1 and Q2 2024?" (original)
2. "What changed Q1"
3. "What changed Q2 2024"
4. "What changed 2024"

**Result:** Documents from both Q1 and Q2 are retrieved and merged, giving the LLM context to explain changes.

## Dependencies

- Requires a chat LLM client for detection (the batched `QueryBatcher` call)
- Part of the combined LLM classification call (no additional latency)

## Related Features

- [**Comparison Queries**](comparison-queries.md) - Multi-entity retrieval (often combined with temporal)
- [**Aggregation Queries**](aggregation-queries.md) - List/count queries (another detection module)
- [**Freshness & Authority**](freshness-authority.md) - Recency boosting (complements temporal filtering)
