# tools/contract_map/

Static analysis tool that enforces fitz-sage's architectural layer dependency rules.

Detects forbidden cross-layer imports at the module level so violations are caught in CI before they reach runtime.

## Rules enforced

- `core/` — no imports from `engines/`, `ingestion/`, `retrieval/`, `llm/`, `storage/`
- `retrieval/`, `llm/`, `ingestion/` — may only import from `core/`
- `engines/` — may import `core/`, `llm/`, `storage/`, `retrieval/`
- `runtime/`, `cli/` — unrestricted

## Usage

```bash
python -m tools.contract_map              # Print violations
python -m tools.contract_map --fail-on-errors   # Exit non-zero on any violation (CI mode)
```
