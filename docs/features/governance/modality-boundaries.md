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
| Unstructured text | Pyrrho evaluates the delivered evidence set directly. | Primary supported governance path. |
| Tables / structured data | KRAG retrieves table metadata and grounded rows, with optional generated SQL when chat tiers are configured; Pyrrho judges the textual evidence representation. | Pyrrho does not prove SQL correctness, join validity, unit conversion, or aggregation completeness. |
| Code | KRAG retrieves symbols, files, imports, and references; Pyrrho judges the retrieved snippets as text. | Pyrrho is not yet trained to prove call-graph completeness, runtime behavior, API compatibility, or test adequacy. |
| Logs / traces / config | Retrieval favors exact tokens and source metadata. | Pyrrho can judge whether shown evidence looks sufficient, but domain-specific failure semantics are not separately modeled. |

## Product Implication

When a query is code-heavy or table-heavy, use Pyrrho's mode as an evidence
sufficiency signal, not as a formal verifier. The caller should still surface
the ranked evidence, source locations, table query metadata, and code/test
context so a developer can inspect the result.

`INSUFFICIENT` is especially useful for these modalities: it means the
fixed delivered evidence did not satisfy the query contract. Fitz-Sage returns
that verdict and evidence unchanged; a caller may prepare better source data or
start a new retrieval with different inputs.

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
