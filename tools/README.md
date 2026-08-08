# tools/

Developer and CI tooling for fitz-sage. Not part of the importable package.

| Directory / File | Purpose |
|---|---|
| `cli_map/` | Static CLI command and wiring checks |
| `contract_map/` | Enforces layer dependency rules across the codebase |
| `pre_release/` | Pre-release lint and verification steps |
| `retrieval_eval/` | Retrieval evaluation helpers and reports |
| `ci_check.py` | CI health checks run in GitHub Actions |
| `mutation_test.py` | Mutation testing runner |
| `pre_release.py` | Entry point for the full pre-release workflow |
| `smoke_local_retrieval.py` | Local point/retrieve/enrichment smoke test |
| `wheel_smoke.py` | Builds or accepts a wheel, installs it in a fresh venv, and smoke-tests packaging/runtime deps |

## Usage

```bash
# Architecture contract check (run before every PR)
python -m tools.contract_map --fail-on-errors

# Pre-release checks
python -m tools.pre_release

# Built wheel smoke
python -m tools.wheel_smoke --smoke import
python -m tools.wheel_smoke --dist-dir dist --skip-build --smoke retrieve

# Mutation testing
python tools/mutation_test.py
```
