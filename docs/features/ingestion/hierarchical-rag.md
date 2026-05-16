# Hierarchical RAG

## Problem

Standard RAG fails on analytical queries because answers are spread across documents:

- **Q:** "What are the design principles?"
- **Standard RAG:** Returns random chunks mentioning "design" or "principles" → fragmented, incomplete
- **Expected:** Aggregated insights spanning all documents

Analytical queries like "What are the trends?", "What are the key themes?", or "Summarize the main points" need **document-level and corpus-level understanding**, not chunk-level retrieval.

## Solution: Multi-Level Summaries

Fitz generates hierarchical summaries during ingestion and retrieves at the appropriate level:

```
Level 2: Corpus summary (all documents)
         ↓
Level 1: Group summaries (per source file)
         ↓
Level 0: Original chunks (granular content)
```

Query routing is automatic — summaries match analytical queries via the BM25 + ONNX cross-encoder reranker pipeline because the summary text matches the broader phrasing those queries use.

## How It Works

### At Ingestion

1. **Level 0: Original chunks** - Documents are chunked normally (500-1000 tokens)
   - Tagged with `hierarchy_level: 0`
   - Enriched with `hierarchy_summary` metadata containing their group's L1 summary

2. **Level 1: Group summaries** - Each source file gets a summary:
   - Aggregates all chunks from that file
   - LLM generates a ~200-word summary
   - **NOT stored as separate chunks** - only as metadata on L0 chunks

3. **Level 2: Corpus summary** - Entire collection gets a summary:
   - Aggregates all Level 1 summaries
   - LLM generates a ~500-word summary
   - **The ONLY separate chunk created** - stored with `hierarchy_level: 2`

### At Query Time

Only L0 and L2 are indexed in FTS5 (L1 lives as metadata on each L0
row). BM25 returns:

```
Q: "What are the overall trends?"
→ L2 corpus summary scores high on the abstract phrasing
→ Returns corpus summary for high-level view
→ Result: High-level answer spanning all documents

Q: "What did users say about the async tutorial?"
→ L0 chunks from async_tutorial.md score high on the specific tokens
→ L1 summary available via hierarchy_summary metadata for context
→ Result: Specific, granular content with file-level context
```

## Key Design Decisions

1. **Always-on** — summaries are generated automatically during ingestion. No configuration needed.

2. **Automatic routing** — abstract queries lexically match the L2
   summary; specific queries match L0 token-for-token. The LLM
   reranker resolves edge cases.

3. **Incremental updates** - When a file changes, only its L1 summary regenerates. L2 regenerates if significant change.

4. **Tagged indexing** - Summaries are stored as special chunks with `hierarchy_level` metadata.

5. **LLM-generated** - Uses the same chat LLM to generate summaries (no separate model).

## Configuration

Minimal configuration required. Feature is baked into the ingestion pipeline.

Optional configuration in `enrichment.yaml`:

```yaml
hierarchy:
  group_by: source_file   # metadata key to group L1 summaries by
```

Internal parameters:
- Group size limits for L1 summaries
- Epistemic assessment for detecting conflicts

## Files

- **Hierarchical enricher:** `fitz_sage/ingestion/enrichment/hierarchy/enricher.py`
- **Chunk grouper (by metadata key):** `fitz_sage/ingestion/enrichment/hierarchy/grouper.py`
- **Summary storage:** L2 indexed in SQLite FTS5 (tagged `hierarchy_level: 2`), L1 as metadata on L0
- **Ingestion hook:** `fitz_sage/ingestion/enrichment/pipeline.py` (calls hierarchy enrichment)

## Benefits

| Standard RAG | Hierarchical RAG |
|--------------|------------------|
| Fragments on analytical queries | Coherent high-level answers |
| No corpus-level view | Automatic corpus summarization |
| Misses themes/trends | Captures themes/trends naturally |
| Only finds direct matches | Finds conceptual matches via summaries |

## Example

**Corpus:** 50 documents about software architecture

### Query: "What are the overall trends?"

**Standard RAG (no hierarchy):**
- Returns: 5 random chunks mentioning "trend"
- Result: Fragmented, incomplete

**Hierarchical RAG:**
- Returns: L2 corpus summary + top L1 file summaries
- Result:

```
The corpus shows three major architectural trends:

1. Microservices adoption: 60% of documents discuss service decomposition,
   API gateways, and inter-service communication patterns.

2. Event-driven design: 40% cover event sourcing, message queues, and
   asynchronous processing.

3. Observability focus: 75% emphasize logging, metrics, and distributed tracing
   as first-class architectural concerns.

Sources: architecture_overview.md (L1), microservices_guide.md (L1),
observability_patterns.md (L1), corpus_summary (L2)
```

### Query: "How do I implement authentication in microservices?"

**Standard RAG (no hierarchy):**
- Returns: 5 chunks directly mentioning authentication
- Result: Granular, specific

**Hierarchical RAG:**
- Returns: L0 chunks (same as standard RAG—no hierarchy needed for specific queries)
- Result: Same as standard RAG

Hierarchy only activates when BM25 + the ONNX cross-encoder reranker promote a summary row over the granular ones.

## When Hierarchy Activates

| Query Type | Retrieved Level | Reason |
|------------|----------------|--------|
| "What are the trends?" | L2 + L1 | Analytical, corpus-level |
| "Summarize the main points" | L2 + L1 | Analytical, corpus-level |
| "What topics are covered?" | L2 + L1 | Meta-question about corpus |
| "How do I authenticate?" | L0 | Specific, granular |
| "What does file X say?" | L1 (X) + L0 (X) | File-specific |

## Dependencies

- Same LLM provider used for answering (no additional dependencies)
- Summaries stored in the same SQLite collection as chunks

## Performance Considerations

- **Ingestion time:** +30-60s per 50 documents (for summary generation)
- **Storage:** +2-5% (summaries are small compared to chunks)
- **Query time:** No additional latency (summaries retrieved like any chunk)

## Related Features

- **Aggregation Queries** - Expands retrieval count; hierarchy provides aggregated content
- **Multi-Query** - Long queries decomposed; hierarchy provides high-level context
- **Epistemic Honesty** - Corpus summary helps detect when information is missing
