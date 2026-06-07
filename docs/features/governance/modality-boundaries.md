<!-- docs/features/governance/modality-boundaries.md -->
# Governance Modality Boundaries

Pyrrho governs whether retrieved evidence is sufficient, disputed, or
insufficient for a query. The current production classifier is strongest on
unstructured text evidence: prose documents, policies, reports, PDFs after
parsing, and ordinary knowledge-base sections.

fitz-sage can retrieve code symbols and structured table evidence, but those
modalities have different sufficiency rules. Treat them as routed evidence
surfaces today, not as separately trained governance domains.

## Current Contract

| Evidence surface | Current behavior | Boundary |
|---|---|---|
| Unstructured text | Pyrrho evaluates evidence prefixes directly. | Primary supported governance path. |
| Tables / structured data | KRAG retrieves table metadata and SQL-backed results; Pyrrho judges the textual evidence representation. | Pyrrho does not prove SQL correctness, join validity, unit conversion, or aggregation completeness. |
| Code | KRAG retrieves symbols, files, imports, and references; Pyrrho judges the retrieved snippets as text. | Pyrrho is not yet trained to prove call-graph completeness, runtime behavior, API compatibility, or test adequacy. |
| Logs / traces / config | Retrieval favors exact tokens and source metadata. | Pyrrho can judge whether shown evidence looks sufficient, but domain-specific failure semantics are not separately modeled. |

## Product Implication

When a query is code-heavy or table-heavy, use Pyrrho's mode as an evidence
sufficiency signal, not as a formal verifier. The caller should still surface
the ranked evidence, source locations, table query metadata, and code/test
context so a developer can inspect the result.

`ABSTAIN` is especially useful for these modalities: it means the retrieved
prefix did not satisfy the query contract, and fitz-sage should either broaden
retrieval or tell the user which source surface appears missing.

## Future Direction

The clean extension is modality-specific governance:

| Future classifier | Evidence it should understand | Extra failure modes |
|---|---|---|
| Pyrrho Structured | schemas, rows, SQL results, units, filters, joins | missing join key, ambiguous aggregation, wrong unit, incomplete group coverage |
| Pyrrho Code | symbols, call sites, tests, docs, logs, diffs | missing caller, stale docs, untested path, version/API mismatch, implementation/spec conflict |

fitz-sage should route by retrieval modality, then apply the classifier whose
training data matches the evidence semantics. Until those classifiers exist,
the current Pyrrho model remains the standard governance layer.

## Related

- [Epistemic Honesty](epistemic-honesty.md)
- [CONSTRAINTS.md](../../CONSTRAINTS.md)
- [Code Symbol Extraction](../ingestion/code-symbol-extraction.md)
- [Tabular Data Routing](../ingestion/tabular-data-routing.md)
