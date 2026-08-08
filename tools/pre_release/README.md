# tools/pre_release/

Pre-release verification steps run before tagging a new version.

| File | Purpose |
|---|---|
| `lint.py` | Runs Black, isort, and type-check passes |
| `prerelease.py` | Orchestrates the full pre-release checklist |

The canonical full workflow is the top-level `tools/pre_release.py`. The files
in this directory are standalone focused alternatives.

## Usage

```bash
python -m tools.pre_release
python tools/pre_release/lint.py --check
python tools/pre_release/prerelease.py --check
```
