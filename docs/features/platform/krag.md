# KRAG Engine

KRAG (Knowledge Routing Augmented Generation) is Fitz-Sage's shipping retrieval
engine. It uses typed source units, sparse recall, structural context, an ONNX
cross-encoder, evidence compilation, and Pyrrho governance. It does not use
dense embeddings or a vector database.

## Source Model

`point()` parses supported inputs into the retrieval unit that matches their
structure:

| Source | Query-ready unit | Search surface |
|---|---|---|
| Prose and rich documents | sections plus stored raw source | section FTS5 |
| Python, TypeScript/JavaScript, Go, Java | symbols and file import edges | symbol FTS5 plus name lookup |
| Configured delimited tables (`.csv`, `.tsv` by default) | table schema and concrete rows | metadata lookup plus row FTS5 |

Embedded tables in documents remain section text. Source bytes and provenance
are retained so retrieval reads the original evidence, not a generated summary
as a substitute.

When `point()` returns, every supported file is searchable or explicitly
reported as failed. Optional Qwen entity, hierarchy, and demand-summary work
runs afterward and never defines query readiness.

## Query Flow

```text
user query
  -> deterministic query shape + Pyrrho PRE obligations
  -> managed Qwen semantic query terms
  -> optional query-intelligence rewrite/analyze/detect
  -> section / symbol / table recall
  -> cross-strategy fusion and exact deduplication
  -> bounded INT8 ONNX reranking
  -> read original source units
  -> bounded evidence closure for unresolved obligations
  -> contract-aware compilation
  -> Pyrrho over ranked prefixes of 3, then +2 while insufficient
  -> EvidencePack
  -> optional synthesizer -> Answer
```

Broad recall is intentionally permissive. Literal terms and Qwen terms compete
inside one bounded pool so alternate vocabulary has room to enter. Reranking
and evidence compilation handle precision; semantic terms do not assert synonym
or identifier equivalence.

## Structural Expansion

After reading recalled addresses, KRAG can add bounded context from:

- parent/child document sections;
- same-file symbol references and class context;
- resolved file import edges;
- optional entity links;
- optional corpus hierarchy for broad queries.

These are retrieval aids, not a complete dependency analyzer or knowledge
graph. The entity graph currently materializes related symbol neighbors; stored
section edges are not a promise of section-neighbor delivery.

## Table Path

Native table recall uses schema terms, concrete row-value BM25, and bounded
deterministic row plans. Exact identifier lookup and supported filters/sorts do
not require a chat model. Configured chat tiers can optionally generate and
validate SQL for one retrieved table.

## Governance Boundary

Pyrrho owns governance. Fitz-Sage uses Pyrrho's query-only PRE heads as evidence
obligations, performs retrieval mechanics, then grows the ranked evidence
prefix only after exact `INSUFFICIENT`. Fitz-Sage mechanically transports every
verdict and does not add local confidence floors, dispute overrides, or fallback
verdicts.

## Public Engine Methods

```python
from pathlib import Path

from fitz_sage import Query, create_engine

engine = create_engine("fitz_krag")
engine.load("company_docs")
engine.point(Path("./docs"), start_worker=False)

pack = engine.evidence(Query(text="Where is the retention policy?"))
raw = engine.retrieve(Query(text="Where is the retention policy?"))
# Requires a configured synthesizer:
answer = engine.answer(Query(text="Where is the retention policy?"))
```

- `evidence()` is the default governed retrieval contract.
- `retrieve()` returns engine-specific `ReadResult` objects.
- `answer()` requires an optional configured synthesizer to write prose.

## Storage

Each collection uses one SQLite database under the current `.fitz/sqlite/`
workspace. Sections, symbols, and row values use FTS5 where appropriate;
ordinary tables hold raw files, schemas, import edges, and entity links. See
[Unified Storage](unified-storage.md).

## Boundaries

- Data cleanup, raw-log compression, OCR selection, private mappings, and
  identifier normalization are user-owned.
- Background summaries and entity links are optional derived metadata.
- Native structured tables are configured delimited files, not arbitrary
  embedded tables or XLSX.
- Retrieval and model budgets are finite; broad or multi-document questions can
  lose coverage.
- Pyrrho's current context and model quality are separate governance limits.

## Implementation

- `fitz_sage/engines/fitz_krag/engine.py`
- `fitz_sage/engines/fitz_krag/query_pipeline.py`
- `fitz_sage/engines/fitz_krag/retrieval/`
- `fitz_sage/engines/fitz_krag/ingestion/`
- `fitz_sage/engines/fitz_krag/progressive/`

## Related

- [Retrieval Pipeline](../../RETRIEVAL_PIPELINE.md)
- [Ingestion Pipeline](../../INGESTION.md)
- [Evidence Pack](../../EVIDENCE_PACK.md)
- [Limitations](../../LIMITATIONS.md)
