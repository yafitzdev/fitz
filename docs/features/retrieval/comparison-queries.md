# Comparison Queries

Comparison handling is a query-shape feature. It attempts to preserve evidence
for each explicit side instead of letting the lexically stronger side consume
the complete result set.

## Detection

The deterministic planner recognizes explicit forms such as:

- `compare X and Y`;
- `X versus Y` or `X vs Y`;
- `difference between X and Y`;
- `what changed between X and Y`.

It extracts sides only when the wording provides a bounded parse. Optional
`query_intelligence` can contribute additional structured detection, but the
standard behavior does not require a chat endpoint.

## Retrieval Behavior

When sides are available and the comparison contract applies, the router keeps
the original query and adds side-focused recall legs:

```text
original comparison query
    + side A with comparison terms
    + side B with comparison terms
    + combined-side query
        -> typed recall and one fused candidate pool
```

Section, symbol, and table strategy calls are parallelized within the configured
retrieval-worker limit. Duplicate addresses merge their query provenance.

After reranking and reading, evidence compilation assigns comparison roles and
tries to retain both sides. Evidence closure may issue bounded follow-up recall
for a missing side before progressive delivery starts.

## Contract

Comparison handling is best effort, not a guarantee that both sides exist or
will fit inside the evidence budget. If one side is absent from the corpus or
never enters recall, Fitz-Sage should expose incomplete evidence and Pyrrho owns
the resulting verdict.

The feature does not infer that two differently written identifiers refer to
the same side.

## Implementation

- `fitz_sage/engines/fitz_krag/query_planner.py`
- `fitz_sage/retrieval/detection/modules/comparison.py`
- `fitz_sage/engines/fitz_krag/retrieval/router.py`
- `fitz_sage/engines/fitz_krag/evidence_compiler.py`
- `fitz_sage/engines/fitz_krag/evidence_closure.py`

## Boundaries

- Implicit or syntactically complex comparisons may not yield explicit sides.
- One side can still dominate when the other is lexically invisible.
- Multi-document comparison remains subject to pointwise reranking and fixed
  delivery budgets.
- Domain equivalence and private mappings remain user-owned.

## Related

- [Temporal Queries](temporal-queries.md)
- [Aggregation Queries](aggregation-queries.md)
- [Limitations](../../LIMITATIONS.md)
