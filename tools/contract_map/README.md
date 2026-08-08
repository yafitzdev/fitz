# tools/contract_map/

Static analysis tool that enforces fitz-sage's architectural layer dependency rules.

Detects forbidden cross-layer imports at the module level so violations are caught in CI before they reach runtime.

## Rules enforced

| Layer | May import from |
|---|---|
| `core` | `core` |
| `encoders` | `encoders` |
| `ingestion` | `core`, `ingestion` |
| `storage` | `core`, `storage` |
| `retrieval` | `core`, `retrieval`, `storage` |
| `llm` | `core`, `encoders`, `llm` |
| `governance` | `core`, `encoders`, `governance` |
| `tabular` | `core`, `llm`, `storage`, `tabular` |
| `config` | `config`, `core` |
| `engines` | `config`, `core`, `engines`, `governance`, `ingestion`, `llm`, `retrieval`, `storage`, `tabular` |
| `api`, `cli`, `runtime`, `sdk`, `services`, `tools` | unrestricted orchestration layers |

## Usage

```bash
python -m tools.contract_map              # Print violations
python -m tools.contract_map --fail-on-errors   # Exit non-zero on any violation (CI mode)
```
