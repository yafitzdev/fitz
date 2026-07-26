<!-- benchmarks/README.md -->
# Retrieval Benchmark

This benchmark is evidence-first. It does not score answer prose. It runs the
real folder-to-evidence path and validates returned evidence against
deterministic expectations.

Compact reports are the default. Each record includes:

- selected evidence identities, without source bodies
- query terms and query-shape signals
- pre-governance ranked evidence identities
- post-governance delivered evidence identities
- candidate counts at recall, reranking, and final-selection boundaries
- governance cutoff trajectory
- stage attribution for missing evidence
- deterministic validation metrics and failures
- aggregate pass-rate summaries by domain and tag
- a per-file ingestion inventory, including unsupported and failed files

Use `--report-detail full` only when source-bearing debug artifacts are needed.
Full reports include the content-bearing `RetrievalRun` and can be large.

## Run

```bash
python -m benchmarks.fitz_bench.runner
```

To run against an unpacked local Pyrrho package:

```bash
python -m benchmarks.fitz_bench.runner --governance "pyrrho/path/to/model"
```

Defaults:

- corpus: `benchmarks/corpora/core`
- cases: `benchmarks/cases/core.yaml`
- JSON report: `benchmarks/results/latest.json`
- Markdown summary: `benchmarks/results/latest.md`
- workspace: `.bench_workspace/<collection>`
- index mode: `complete`
- report detail: `compact`
- gate: all assertions, including governance

For a quick smoke run:

```bash
python -m benchmarks.fitz_bench.runner --limit 2 --index-mode progressive
```

Generated reports under `benchmarks/results/` and benchmark workspaces under
`.bench_workspace/` are ignored by git.

## Case Shape

```yaml
- id: code_symbol_lookup
  domain: code
  query: "Where is expired session refresh implemented?"
  expected:
    mode: sufficient
    required_evidence:
      - file: code/auth_service.py
        kind: symbol
        location_contains: "refresh_expired_session"
        contains: ["refresh_expired_session", "grace"]
    forbidden_evidence:
      - file: docs/stale.md
        contains: ["old behavior"]
  tags: [code, symbol_lookup]
```

Cases use the v2 evidence-verdict names: `sufficient`, `disputed`, and
`insufficient`.

The validator checks evidence items, not generated text. A required evidence
entry passes when any returned evidence item matches the file/kind/location and
contains every listed text fragment.

## Suites

The starter corpus covers:

- unstructured prose
- structured CSV/table evidence
- code symbols
- mixed table/prose cases
- conflicts and insufficient evidence
- temporal freshness and stale evidence
- explicit in-corpus acronym bridges
- filtered table lookups and comparisons
- stale documentation versus implementation conflicts
- cross-domain multi-hop retrieval

Available suites:

- Core: `benchmarks/corpora/core` with `benchmarks/cases/core.yaml`
- Holdout: `benchmarks/corpora/holdout` with `benchmarks/cases/holdout.yaml`
- Holdout2: `benchmarks/corpora/holdout2` with `benchmarks/cases/holdout2.yaml`

Run Holdout2 explicitly:

```bash
python -m benchmarks.fitz_bench.runner --corpus benchmarks/corpora/holdout2 --cases benchmarks/cases/holdout2.yaml --output benchmarks/results/holdout2_latest.json --markdown benchmarks/results/holdout2_latest.md
```

Grow this by adding more files under `benchmarks/corpora/` and YAML cases under
`benchmarks/cases/`.

For focused diagnosis, repeat `--case-id` without editing the suite:

```bash
python -m benchmarks.fitz_bench.runner \
  --corpus benchmarks/corpora/limits \
  --cases benchmarks/cases/limits.yaml \
  --case-id structured_large_rec0619_owner \
  --case-id conflict_run55b_final_audit
```

The runner prints one progress line per completed case. The full limitations
suite exercises managed Qwen, reranking, evidence closure, and Pyrrho for every
query, so it is a release-gate run rather than a fast smoke test.

## Production Matrix

The production matrix composes standard, holdout, corpus-growth, format,
reload-stability, optional-format, and intentional-limitations suites:

```bash
python -m benchmarks.fitz_bench.production_runner \
  --governance "pyrrho/C:\path\to\reviewed-clean-onnx-package"
```

The required matrix currently contains:

- core, holdout, and second holdout corpora
- an 80-document near-neighbor corpus-growth run
- reload stability over the grown corpus
- 60 explicit temporal, comparison, aggregation, and narrow query-shape cases
- PDF, DOCX, PPTX, SQL, Python, Go, Java, and TypeScript evidence
- measured, non-gating XLSX and known-limitations suites

Required retrieval suites gate at 85%. The query-shape suite gates its own
signals at 85%. Corpus growth fails the production gate if a shared case
regresses even when the aggregate rate remains above threshold. Governance is
measured separately because Pyrrho has its own fixed-evidence evaluation and
release lifecycle. Supported-file ingestion failures always fail a required
suite.

Run one suite while developing:

```bash
python -m benchmarks.fitz_bench.production_runner \
  --suite-id base_formats \
  --governance "pyrrho/C:\path\to\reviewed-clean-onnx-package"
```

## Metric Boundaries

- `retrieval_pass_rate`: required evidence is present and forbidden evidence is
  absent in the compiled ranking before Pyrrho cutoff.
- `delivery_pass_rate`: the same evidence assertions over the final governed
  `EvidencePack`.
- `query_shape_pass_rate`: explicit query-plan signals match their human-labeled
  temporal, comparison, aggregation, or narrow expectation.
- `capability_pass_rate`: all evaluated pre-governance retrieval and
  query-shape assertions pass.
- `pass_rate`: governed delivery, query shape, and expected Pyrrho mode all
  pass.
- `retrieval_stability_rate`: pre-governance ranked identities survive a fresh
  engine load.
- `delivery_stability_rate`: governed evidence identities survive a fresh
  engine load.
- `governance_stability_rate`: the Pyrrho mode survives a fresh engine load.

Every rate includes an `*_evaluated` denominator. Cases without an assertion
for that metric do not receive automatic credit. Do not use the full pass rate
to describe retrieval quality. A mode-only Pyrrho error is reported as a
governance failure; a correct item removed by cutoff is a delivery failure, not
a recall failure.

## Balanced Governance

The retrieval suites above are product/integration benchmarks. They are not
class-balanced: most cases are expected to be sufficient. To compare Pyrrho
governance models directly, use the fixed-evidence balanced benchmark:

```bash
python -m benchmarks.fitz_bench.governance_runner --governance "pyrrho/C:\path\to\pyrrho\best_model" --output benchmarks/results/governance_balanced_model.json --markdown benchmarks/results/governance_balanced_model.md
```

This suite bypasses live retrieval and feeds Pyrrho 120 fixed evidence packs:
40 sufficient, 40 disputed, and 40 insufficient. It reports accuracy, macro
recall, per-class recall, false-sufficient rate, and false-reject-sufficient
rate.
