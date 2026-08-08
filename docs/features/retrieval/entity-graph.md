# Entity Graph

The entity graph is optional background metadata for finding source units that
mention a shared named entity. It is not required for source indexing or the
first query.

## Population

During background enrichment:

- code-symbol entities and temporal metadata are proposed by managed Qwen;
- document-section entities are derived deterministically from section text or
  an available summary;
- entity-to-unit edges are stored in the collection database.

Tables are not currently populated into this graph. Fitz-Sage does not use the
graph to create domain aliases or to equate differently spelled identifiers.

## Schema

```sql
CREATE TABLE entities (
    name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    entity_type TEXT,
    mention_count INTEGER DEFAULT 1
);

CREATE TABLE entity_chunks (
    entity_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    PRIMARY KEY (entity_name, chunk_id)
);
```

The `chunk_id` column stores a symbol or section identifier. These
are ordinary SQLite tables with indexes, not FTS5 tables or a separate graph
database.

Entity names are lowercased and trimmed for graph keys. That internal graph-key
normalization does not rewrite source text or declare separator/abbreviation
variants equivalent.

## Expansion

For already-read symbol or section IDs, the store can rank neighboring IDs by
the number of shared entities. The KRAG expander currently materializes bounded
related **symbol** neighbors into evidence. Section edges are persisted, but
section-neighbor materialization is not part of the current retrieval contract.

```text
recalled source units
    -> collect their entity edges
    -> rank related IDs by shared entities
    -> materialize unseen related symbols within the profile budget
```

This is context expansion after lexical recall, not a replacement for BM25 and
not a guarantee that every source mentioning an entity is delivered.

## Failure Behavior

An entity-stage failure is reported in `indexing_status().enrichment`. The
original sections, symbols, tables, and FTS5 indexes remain available. A later
enrichment continuation can retry the failed stage.

## Implementation

- `fitz_sage/retrieval/entity_graph/store.py`
- `fitz_sage/engines/fitz_krag/ingestion/enricher.py`
- `fitz_sage/engines/fitz_krag/ingestion/pipeline.py`
- `fitz_sage/engines/fitz_krag/retrieval/expander.py`

## Boundaries

- Entity extraction can omit or misclassify names.
- Shared names do not prove a semantic relationship.
- Tables do not contribute graph edges.
- Current retrieval materializes related symbols, not related sections.

## Related

- [Background Enrichment](../../ENRICHMENT.md)
- [Code Symbol Extraction](../ingestion/code-symbol-extraction.md)
- [Limitations](../../LIMITATIONS.md)
