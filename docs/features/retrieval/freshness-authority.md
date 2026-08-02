# Freshness Intent

## Problem

Queries such as "What is the latest status?" require temporal evidence
handling. The time a file happened to be indexed does not establish when its
content became effective or which source is authoritative.

## Behavior

`FreshnessModule` detects current/recent query wording as query shape:

- `latest`
- `recent`
- `current`
- `updated`
- `newest`
- `today`
- `fresh`

That signal contributes to the temporal retrieval contract. Fitz-Sage can then
prefer evidence whose content expresses the requested scope, such as explicit
dates or `current`/`final` wording, while retaining competing historical
evidence for Pyrrho.

Fitz-Sage does not boost a document from its ingestion timestamp, filesystem
modification time, or row-update time. Users must put effective dates or
authority markers in the source material when those distinctions matter.

## Flow

```text
Query asks for current/recent evidence
    |
    v
Freshness intent is detected
    |
    v
Temporal query contract is built
    |
    v
Content-grounded temporal evidence is ordered for Pyrrho
```

## Implementation

- Detection module: `fitz_sage/retrieval/detection/modules/freshness.py`
- Deterministic planning: `fitz_sage/engines/fitz_krag/query_planner.py`
- Temporal evidence ordering: `fitz_sage/engines/fitz_krag/evidence_compiler.py`

The standard planner detects these terms deterministically. The optional
query-intelligence module can parse the same signal from a batched provider
response. Neither path produces a filesystem-recency score multiplier.

## Related Features

- [Temporal Queries](temporal-queries.md)
- [Aggregation Queries](aggregation-queries.md)
- [Comparison Queries](comparison-queries.md)
