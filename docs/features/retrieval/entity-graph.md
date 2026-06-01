<!-- docs/features/retrieval/entity-graph.md -->
# Entity Graph (Related-Address Discovery)

## Problem

Token-overlap retrieval (FTS5 BM25) returns rows independently. That
fails when:

- **Q:** "What else mentions TechCorp?"
- **Pure BM25:** returns only rows that match the query terms
- **Expected:** also return rows that mention the same entities
  (TechCorp's products, people, related systems)

BM25 doesn't model entity relationships. If section A and section B
both mention `AuthService` but discuss different aspects, they won't
co-occur unless the query happens to match both.

## Solution: entity-based linking

During ingestion, extract named entities per typed unit (symbol /
section / table) and store them as edges in a small SQLite graph.
At query time, after the BM25 hit list comes back, expand it by
walking shared-entity edges.

```
Initial BM25 hits:  [Section A (mentions AuthService, OAuth2)]
                              │
                              ▼
                       Entity-graph lookup
                              │
                              ▼
                   Shared entities: AuthService, OAuth2
                              │
                              ▼
              Other addresses mentioning the same entities
                              │
                              ▼
Expanded set:   [Section A, Section B (AuthService), Section C (OAuth2)]
```

## How it works

### At ingestion time

1. **Entity extraction.** `KragEnricher` extracts named entities from
   each typed unit (symbol / section) via the chat-tier LLM, as part of
   its batched keyword + entity + temporal extraction.
2. **Graph population.** During the pipeline's `enrich` step, the
   extracted entities + unit associations are written to the
   `EntityGraphStore` — two SQLite tables alongside the rest of the
   collection:

   ```
   entities         : (id, name, type, mention_count)
   entity_addresses : (entity_id, address_id, count)
   ```

   FTS5 index on `entities.name` lets you do prefix / phrase lookups
   on entity names without scanning.

### At query time

1. **Initial retrieval.** BM25 + KRAG routing returns a top-K set.
2. **Entity lookup.** Pull entities for the top hits.
3. **Graph expansion.** Find other addresses sharing those entities,
   subject to `min_shared_entities` and `max_total`.
4. **Ranking.** Sort expansions by number of shared entities.
5. **Merge + dedupe.** Combine with the original BM25 hits.

```python
initial = bm25_search(query, k=20)
ents    = entity_store.entities_for(initial)
related = entity_store.addresses_sharing(ents, min_shared=2, max_total=10)
final   = dedupe(initial + related)
```

## Key design decisions

1. **Always-on.** Baked into the enrichment + retrieval pipelines;
   no configuration knob.
2. **SQLite-native storage.** Lives alongside the rest of the
   collection in the same `.db` file. No separate graph database, no
   network.
3. **Lightweight graph.** Only stores entity-address edges + a
   denormalised mention count. No full entity attributes.
4. **Ingestion-time extraction.** Entities extracted once via the
   chat model during ingest; query-time path is pure SQL.
5. **Configurable expansion.** `min_shared_entities` controls how
   tight the linkage must be (default 1); `max_total` caps the size
   of the expanded set.

## Entity types

`KragEnricher` extracts these entity types (tunable via prompt):

| Type           | Examples                                          |
| -------------- | ------------------------------------------------- |
| `class`        | `AuthService`, `UserController`, `PaymentGateway` |
| `function`     | `validateToken()`, `processPayment()`             |
| `person`       | John Smith, Alice (when contextually a person)    |
| `organization` | TechCorp, Acme Inc, Engineering Team              |
| `technology`   | OAuth2, SQLite, React, Kubernetes                 |
| `concept`      | Authentication, Rate Limiting, Caching            |

## Configuration

None for the user. Tunable via constructor / engine config:

- `max_total` — max related addresses retrieved (default 20)
- `min_shared_entities` — minimum shared entities to count as related
  (default 1)

## Files

| Component             | Path                                                              |
| --------------------- | ----------------------------------------------------------------- |
| Graph store           | `fitz_sage/retrieval/entity_graph/store.py` (`EntityGraphStore`)  |
| Entity extraction     | `fitz_sage/engines/fitz_krag/ingestion/enricher.py` (`KragEnricher`) |
| Graph population      | `fitz_sage/engines/fitz_krag/ingestion/pipeline.py` (`_populate_entity_graph`) |
| KRAG integration      | `fitz_sage/engines/fitz_krag/retrieval/expander.py`               |

## Example

**Sections:**
- A: "The `AuthService` class validates JWT tokens using the OAuth2 protocol."
- B: "`AuthService` logs all authentication attempts to the audit table."
- C: "OAuth2 refresh tokens expire after 30 days by default."

**Query:** "How does authentication work?"

**Without entity graph:** BM25 returns A (strongest token overlap).

**With entity graph:**
- Initial: A
- Entities found in A: `AuthService`, `OAuth2`
- Graph expansion: B (shares `AuthService`), C (shares `OAuth2`)
- Final: A, B, C

The synthesizer then has complete context about `AuthService`
behaviour *and* OAuth2 configuration.

## Dependencies

- `KragEnricher` entity extraction during ingestion (runs when `enricher:` is
  configured).
- Same SQLite `.db` as the rest of the collection (no separate store).
- Runs inside the KRAG expander; no extra LLM calls at query time.

## Related

- **Enrichment** — extracts the entities during ingest
- [Multi-Hop Reasoning](multi-hop-reasoning.md) — can use the entity
  graph for bridge extraction
- [Comparison Queries](comparison-queries.md) — entity graph helps
  retrieve both compared entities
