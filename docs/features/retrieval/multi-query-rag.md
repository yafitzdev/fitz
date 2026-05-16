# Multi-Query RAG

## Problem

Standard RAG takes a large query (e.g., a full test report + spec
excerpt + requirements doc) and uses it as one BM25 query. The TF
distribution gets flat, the relevant tokens drown in surrounding
context, and FTS5 returns mostly noise.

## Solution: automatic query decomposition

Instead of running the entire input as one query, ask the fast chat
tier to extract key search terms, then run multiple targeted FTS5
queries.

## How it works

```
Query comes in
    │
    ├─ len(query) < 300 chars? → Single FTS5 + bm25() search
    │
    └─ len(query) >= 300 chars?
           │
           ▼
       Fast chat tier: "Extract key search terms"
           │
           ▼
       3–5 targeted sub-queries
           │
           ▼
       FTS5 + bm25() per sub-query  →  Dedupe  →  Rerank  →  Return
```

## Key Design Decisions

1. **Always-on** - No user configuration needed. Built into the KRAG retrieval pipeline.

2. **Fast LLM** - Uses `tier="fast"` model for query expansion. Cheap (~100-200ms) and negligible cost.

3. **Length-based routing** - Only triggers for queries ≥300 characters. Short queries bypass expansion entirely.

4. **LLM handles extraction** - No regex, no entity configuration. LLM figures out what's important (Jira tickets, error codes, names, etc.).

5. **Graceful degradation** - If chat client unavailable, falls back to single search.

## Example

**Input (long test report):**
```
Test TC_CAN_001 failed with error 0x4F on CAN Bus module.
The test was checking timeout behavior and got "no response" after 500ms.
Expected: ACK within 100ms. Actual: Timeout.
...
```

**LLM extracts:**
```json
["TC_CAN_001 known issues", "CAN Bus timeout", "error 0x4F", "no response timeout"]
```

**Result:** 4 targeted searches instead of 1 diluted search. Better retrieval precision.

## Configuration

No configuration required — multi-query expansion is automatic in the KRAG retrieval pipeline. Queries longer than ~300 characters are expanded; shorter queries are not.

## Benefits

| Standard RAG | Multi-Query RAG |
|--------------|-----------------|
| 1 diluted query | N targeted queries |
| Random chunks | Relevant chunks |
| LLM filters noise | LLM analyzes |
| ~30-50% precision | ~60-80% precision |

## Use Cases

- Test failure analysis (automotive, QA)
- Log analysis with structured data
- Support ticket routing with metadata
- Any scenario with long, structured input + unstructured knowledge base

## Related

- [Query Expansion](query-expansion.md) — rule-based synonym/acronym
  expansion; complements multi-query
- [Sparse Search (FTS5 + bm25)](sparse-search.md) — what each
  sub-query actually hits
- [Hierarchical RAG](../ingestion/hierarchical-rag.md) — summaries
  help when multi-query extracts high-level themes
- [Reranking](reranking.md) — the ONNX cross-encoder step that follows
  the merged sub-query results
