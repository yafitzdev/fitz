# Aggregation And Set Queries

Queries such as "list all failed tests" or "how many incidents" need broader
coverage than a narrow fact lookup. Fitz-Sage recognizes this as query shape and
adjusts retrieval; it does not claim database-style exhaustiveness over arbitrary
documents.

## Detection

The deterministic planner recognizes bounded forms including:

- `how many`, `count`, and `number of`;
- `list all`, `enumerate`, `every`, and `complete inventory`;
- `unique` and `distinct`;
- plural set questions such as `which services are affected`.

Scalar measurements such as "how many seconds is the timeout" are excluded
from count intent. Optional `query_intelligence` can add analysis, but the
standard detector does not require an endpoint.

## Retrieval Behavior

Aggregation intent produces a broad/exhaustive coverage profile. Compared with
a narrow lookup, that profile can:

- collect and read a larger candidate set;
- keep relevant modalities eligible;
- inject an available corpus overview for broad context;
- use evidence closure for unresolved coverage obligations;
- compile a more representative ranked prefix.

The original query remains in recall. There is no generated instruction text
silently appended to the user's query.

## Tables

For native CSV/TSV tables, exact identifier lookup and supported deterministic
filter/sort plans can inspect concrete rows. That is stronger than prose-set
retrieval, but it still has configured result limits and does not provide
multi-table join orchestration.

## Contract

An aggregation-shaped `EvidencePack` is evidence for a set answer, not proof
that every matching item in an arbitrary corpus was found. Completeness depends
on source structure, vocabulary, candidate budgets, table limits, and whether
the corpus itself expresses the requested category consistently.

Pyrrho evaluates the exact delivered set. Fitz-Sage does not override its final
verdict with a deterministic "complete" claim.

## Implementation

- `fitz_sage/engines/fitz_krag/query_planner.py`
- `fitz_sage/retrieval/detection/modules/aggregation.py`
- `fitz_sage/engines/fitz_krag/retrieval_profile.py`
- `fitz_sage/engines/fitz_krag/evidence_closure.py`
- `fitz_sage/engines/fitz_krag/evidence_compiler.py`

## Boundaries

- Prose enumeration is bounded and best effort.
- Inconsistent category names can split one logical set.
- Counts over incomplete evidence are not certified as corpus totals.
- Native table handling is limited to configured delimited files and supported
  deterministic or optional SQL operations.

## Related

- [Comparison Queries](comparison-queries.md)
- [Hierarchy Summaries](../ingestion/hierarchical-rag.md)
- [Native Table Routing](../ingestion/tabular-data-routing.md)
- [Limitations](../../LIMITATIONS.md)
