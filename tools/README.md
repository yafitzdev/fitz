# tools/

Developer and CI tooling for fitz-sage. Not part of the importable package.

| Directory / File | Purpose |
|---|---|
| `contract_map/` | Enforces layer dependency rules across the codebase |
| `detection/` | Training and evaluation scripts for the detection classifier |
| `governance/` | Governance pipeline tooling |
| `cli_map/` | Generates CLI command documentation |
| `pre_release/` | Pre-release lint and verification steps |
| `ci_check.py` | CI health checks run in GitHub Actions |
| `mutation_test.py` | Mutation testing runner |
| `pre_release.py` | Entry point for the full pre-release workflow |
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
