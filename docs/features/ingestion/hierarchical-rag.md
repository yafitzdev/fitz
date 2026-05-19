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
Level 0: Original sections (granular content)
```

Query routing is automatic — summaries match analytical queries via the BM25 + ONNX cross-encoder reranker pipeline because the summary text matches the broader phrasing those queries use.

## How It Works

The `KragIngestPipeline` builds the hierarchy itself — there is no
separate hierarchy enricher. L1 summaries are produced during the
per-file `enrich` step; the L2 summary during the corpus `finalize`
step. Both are gated by the `enable_hierarchy` config flag.

### At Ingestion

1. **Level 0: Original sections** - Documents are parsed into sections
   normally, with 1-2 sentence per-section LLM summaries.

2. **Level 1: Group summaries** - During `enrich_file`, each document
   file gets one group summary:
   - The pipeline summarizes the file's sections into a 2-3 sentence
     overview of what the document covers.
   - **NOT stored as a separate section** - written to each section's
     `metadata["hierarchy_summary"]`.
   - Code symbols carry their own machine-readable structure (imports,
     AST), so they get no hierarchy summary.

3. **Level 2: Corpus summary** - During `finalize`, the entire
   collection gets a summary:
   - The pipeline rolls all L1 summaries up into a 3-5 sentence corpus
     summary describing the overall system.
   - **Stored as a synthetic retrievable section** titled "Corpus
     Overview" under a fixed synthetic raw file, so re-ingest upserts
     it in place.

### At Query Time

Only L0 sections and the L2 corpus section are indexed in FTS5 (L1
lives as metadata on each L0 row). BM25 returns:

```
Q: "What are the overall trends?"
→ L2 corpus summary scores high on the abstract phrasing
→ Returns corpus summary for high-level view
→ Result: High-level answer spanning all documents

Q: "What did users say about the async tutorial?"
→ L0 sections from async_tutorial.md score high on the specific tokens
→ Result: Specific, granular content from the matching file
```

## Key Design Decisions

1. **On by default** — summaries are generated automatically during
   ingestion, gated by the `enable_hierarchy` config flag.

2. **Automatic routing** — abstract queries lexically match the L2
   summary; specific queries match L0 token-for-token. The LLM
   reranker resolves edge cases.

3. **Wholesale rebuild** - `finalize` re-runs the L2 summary on every
   re-ingest. Incremental hierarchy is a v2 concern.

4. **Two storage shapes** - L1 lives as `hierarchy_summary` metadata on
   each L0 section; L2 is a single synthetic retrievable section.

5. **LLM-generated** - Uses the same chat LLM to generate summaries (no separate model).

## Configuration

The feature is built into the KRAG ingestion pipeline and controlled by
one engine-config flag:

```yaml
enable_hierarchy: true   # build L1/L2 summaries during ingestion (default)
```

`enable_hierarchy` is independent of `enable_enrichment` — a document
file still gets an L1 summary when keyword/entity enrichment is off.

## Files

- **Ingestion pipeline:** `fitz_sage/engines/fitz_krag/ingestion/pipeline.py`
  — `KragIngestPipeline` builds L1 (`_generate_l1_summary`) during
  `enrich_file` and L2 (`_build_corpus_summary`) during `finalize`
- **Summary storage:** L2 stored as a synthetic "Corpus Overview"
  section indexed in SQLite FTS5; L1 as `hierarchy_summary` metadata on
  each L0 section
- **Config flag:** `enable_hierarchy` in `fitz_sage/engines/fitz_krag/config/schema.py`

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
- Returns: L2 corpus summary
- Result:

```
The corpus shows three major architectural trends:

1. Microservices adoption: 60% of documents discuss service decomposition,
   API gateways, and inter-service communication patterns.

2. Event-driven design: 40% cover event sourcing, message queues, and
   asynchronous processing.

3. Observability focus: 75% emphasize logging, metrics, and distributed tracing
   as first-class architectural concerns.

Sources: Corpus Overview (L2 corpus summary)
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
| "What are the trends?" | L2 | Analytical, corpus-level |
| "Summarize the main points" | L2 | Analytical, corpus-level |
| "What topics are covered?" | L2 | Meta-question about corpus |
| "How do I authenticate?" | L0 | Specific, granular |
| "What does file X say?" | L0 (X) | File-specific |

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
