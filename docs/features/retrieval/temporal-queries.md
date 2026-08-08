# Temporal Queries

Fitz-Sage treats explicit time scope as query shape. The standard path detects
it deterministically; an optional `query_intelligence` provider can add richer
analysis but is not required.

## Detection

The deterministic planner recognizes content such as:

- quarters, months, and standalone years;
- `last month`, `next quarter`, `today`, and similar relative periods;
- `latest`, `current`, `final`, `previous`, and `superseded`;
- `since`, `before`, `after`, and `as of` scopes;
- launch, release, deployment, migration, incident, and outage wording.

This is bounded pattern recognition, not a complete natural-language date
parser.

## Retrieval Behavior

The original query always remains a recall leg. When at least two temporal
references are explicit, the planner can add period-focused query legs and tag
their candidates with the reference that surfaced them.

```text
"What changed between Q1 and Q2?"
    -> original query
    -> Q1-focused query
    -> Q2-focused query
```

The retrieval profile marks temporal intent and the evidence compiler prefers
source spans that explicitly express current/final/latest scope when the
temporal contract calls for it. Competing historical evidence remains visible
for Pyrrho.

## No Filesystem Recency

Fitz-Sage does not infer authority from:

- ingestion order;
- filesystem modification time;
- SQLite row update time;
- a version-looking identifier by itself.

Dates, effective scope, or authority markers must be present in source content
when they matter. Pyrrho decides whether the delivered evidence has the right
scope or remains disputed/insufficient.

## Boundaries

- One source can mix historical and current facts in the same section.
- A version token may not be chronological.
- Period-focused recall is best effort within finite candidate budgets.
- Missing source scope cannot be repaired from file metadata.

## Implementation

- `fitz_sage/engines/fitz_krag/query_planner.py`
- `fitz_sage/engines/fitz_krag/retrieval/router.py`
- `fitz_sage/engines/fitz_krag/evidence_compiler.py`
- `fitz_sage/retrieval/detection/modules/temporal.py`

## Related

- [Freshness Intent](freshness-authority.md)
- [Comparison Queries](comparison-queries.md)
- [Evidence Signals](evidence-signals.md)
