<!-- benchmarks/README.md -->
# Retrieval Benchmark

This benchmark is evidence-first. It does not score answer prose. It runs
fitz-sage retrieval, stores the full `EvidencePack`, and validates the returned
evidence against deterministic expectations.

The JSON report is also the debug artifact. Each record includes:

- the public `EvidencePack`
- query-profile signals
- retrieval trace: strategy calls, candidate frontier, reranker input/output,
  final read set, retry traces
- evidence compiler trace: Pyrrho query contract, Pyrrho-required modalities,
  literal anchors, evidence roles, minimum source count
- governance cutoff metadata, including Pyrrho prefix trajectory
- deterministic validation metrics and failures
- aggregate pass-rate summaries by domain and tag

## Run

```bash
python -m benchmarks.fitz_bench.runner
```

Defaults:

- corpus: `benchmarks/corpora/core`
- cases: `benchmarks/cases/core.yaml`
- JSON report: `benchmarks/results/latest.json`
- Markdown summary: `benchmarks/results/latest.md`
- index mode: `complete`

For a quick smoke run:

```bash
python -m benchmarks.fitz_bench.runner --limit 2 --index-mode progressive
```

Generated reports under `benchmarks/results/` are ignored by git.

## Case Shape

```yaml
- id: code_symbol_lookup
  domain: code
  query: "Where is expired session refresh implemented?"
  expected:
    mode: trustworthy
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

The validator checks evidence items, not generated text. A required evidence
entry passes when any returned evidence item matches the file/kind/location and
contains every listed text fragment.

## Suites

The starter corpus covers:

- unstructured prose
- structured CSV/table evidence
- code symbols
- mixed table/prose cases
- conflicts and abstention
- temporal freshness and stale evidence
- acronym expansion
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
