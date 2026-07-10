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
- evidence compiler trace: Pyrrho v2 verdict, failure, intent, and
  evidence-kind metadata, literal anchors, evidence roles, minimum source count
- governance cutoff metadata, including Pyrrho prefix trajectory
- deterministic validation metrics and failures
- aggregate pass-rate summaries by domain and tag

## Run

```bash
python -m benchmarks.fitz_bench.runner
```

To run against an unpacked local Pyrrho package:

```bash
python -m benchmarks.fitz_bench.runner --governance "pyrrho/C:\Users\yanfi\PycharmProjects\pyrrho\outputs\modernbert_base_v2_alpha\best_model"
```

Defaults:

- corpus: `benchmarks/corpora/core`
- cases: `benchmarks/cases/core.yaml`
- JSON report: `benchmarks/results/latest.json`
- Markdown summary: `benchmarks/results/latest.md`
- workspace: `.bench_workspace/<collection>`
- index mode: `complete`

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
