# Hierarchy Summaries

Hierarchy summaries are optional background context for broad questions. They
do not participate in the `point()` query-readiness boundary.

## Levels

| Level | Stored form | Availability |
|---|---|---|
| L0 | Original document sections | Immediately after source indexing |
| L1 | One file overview copied into section metadata | After background file enrichment |
| L2 | Synthetic `Corpus Overview` section | After background corpus finalization |

Code symbols and native CSV/TSV tables do not receive the document hierarchy.
Their structure is represented by symbol/import metadata and table schemas.

## Lifecycle

```text
parse and store original sections
    |
    +--> point() returns: L0 is searchable
    |
    +--> background build_hierarchy_file(): L1 file overview
    |
    +--> background build_corpus_hierarchy(): L2 corpus overview
```

The managed local Qwen runtime writes L1 and L2 summaries. A model failure is
recorded in enrichment status and leaves the original L0 index intact.

Demand summaries are separate: after retrieval surfaces a document or table,
the warm loop may summarize that source for later reranking. Unqueried sources
are not eagerly summarized merely to make the collection searchable.

## Query Behavior

L1 overviews remain metadata on their original sections. The L2 corpus overview
is persisted as a synthetic section, but it is excluded from ordinary section
BM25. The retrieval profile injects it only for broad or exploratory queries
when a compatible summary exists.

This avoids allowing a generic corpus summary to compete with precise source
sections on every lookup. It also means hierarchy is best-effort context, not a
guarantee that an aggregation or trend question is complete.

## Regeneration

Corpus finalization replaces the schema-versioned synthetic overview rather
than accumulating old summaries. Changed files return to pending enrichment;
unchanged searchable source remains available throughout the process.

## Configuration

There is no public hierarchy provider or enable flag. Source indexing always
works without hierarchy. The in-process worker or CLI enrichment daemon runs
the managed background stages when work is pending.

Use `indexing_status()` to distinguish the two contracts:

```python
status = engine.indexing_status()
print(status["query_ready"])
print(status["enrichment"]["complete"])
```

## Implementation

- `fitz_sage/engines/fitz_krag/ingestion/pipeline.py`
  - `build_hierarchy_file()`
  - `build_corpus_hierarchy()`
  - `summarize_file()`
- `fitz_sage/engines/fitz_krag/progressive/worker.py`
- `fitz_sage/engines/fitz_krag/ingestion/section_store.py`

## Boundaries

- Summaries can omit details or flatten disagreements.
- L2 is representative context, not proof of exhaustive corpus coverage.
- Background completion time depends on corpus size and local Qwen runtime.
- Original source evidence remains authoritative; Pyrrho receives selected raw
  source units, not a hierarchy-only answer.

## Related

- [Ingestion Pipeline](../../INGESTION.md)
- [Background Enrichment](../../ENRICHMENT.md)
- [Aggregation Queries](../retrieval/aggregation-queries.md)
