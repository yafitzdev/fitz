# Contributing To Fitz-Sage

Keep contributions focused, testable, and aligned with the public retrieval
contract. Be respectful and constructive in issues and reviews.

## Development Setup

```bash
git clone https://github.com/yafitzdev/fitz-sage.git
cd fitz-sage

python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest tests/unit/
python -m tools.contract_map --fail-on-errors
```

## Architecture Rules

```text
fitz_sage/
├── core/              # stable types and protocols
├── engines/fitz_krag/ # shipping retrieval engine
├── retrieval/         # shared query/graph helpers
├── ingestion/         # built-in parser/source implementations
├── llm/               # local models and optional endpoint providers
├── storage/           # SQLite connection management
├── tabular/           # native CSV/TSV storage
├── integrations/      # Pyrrho boundary
├── runtime/           # engine registry and runner
├── cli/               # command line
├── api/               # REST API
└── sdk/               # stateful Python API
```

The enforced dependency direction is defined in
`tools/contract_map/architecture.py`. In brief, foundation layers have narrow
allowlists, engines compose the shared layers, and API/CLI/runtime/SDK/service
modules are orchestration surfaces. Do not duplicate a different dependency
table in code or documentation.

Run `python -m tools.contract_map --fail-on-errors` for every architecture
change.

## Product Boundaries

Changes must preserve these decisions unless a proposal explicitly changes the
product contract:

- `point()` completes the searchable source index before returning and does not
  load Qwen.
- background entity and hierarchy work is optional and independently reported.
- BM25 over typed source units is the central recall mechanism; no dense index
  exists.
- broad competition between literal and Qwen candidates is intentional.
- domain cleanup, private mappings, and identifier normalization are user-owned.
- temporal/comparison/aggregation recognition is package-owned query shape.
- Pyrrho owns governance; Fitz-Sage transports its PRE obligations and final
  decision without local verdict heuristics.

Read [Architecture](docs/ARCHITECTURE.md),
[Retrieval Pipeline](docs/RETRIEVAL_PIPELINE.md), and
[Limitations](docs/LIMITATIONS.md) before changing shared behavior.

## Making A Change

1. Reproduce the issue with the smallest relevant test or benchmark case.
2. Identify the earliest failing stage: source index, recall, rerank/read,
   closure/compiler, delivery, or Pyrrho.
3. Implement a general fix at the owning boundary.
4. Add focused tests proportional to the blast radius.
5. Update current documentation when a public contract changes.
6. Run formatting, tests, and the contract map.

Do not add case-specific cleanup, hidden alias rules, compatibility shims, or
Fitz-side governance safeguards to make a benchmark green.

## Engines And Extensions

The minimum engine protocol is structural:

```python
from fitz_sage.core import Answer, Query


class MyEngine:
    def answer(self, query: Query) -> Answer:
        return Answer(text="application-owned answer")
```

The in-process registry accepts custom factories. The installed package does
not discover third-party entry points; automatic discovery covers only bundled
directories under `fitz_sage/engines/`. See
[Custom Engines](docs/CUSTOM_ENGINES.md).

Other extension boundaries are package contributions:

- chat/vision/rerank providers implement the relevant protocol under
  `fitz_sage/llm/providers/` and are dispatched in `fitz_sage/llm/config.py`;
- parser modes are wired explicitly through
  `fitz_sage/ingestion/parser/router.py`;
- KRAG typed-unit extraction is internal, not a generic chunker API.

See [Extension Points](docs/PLUGINS.md).

## Testing

Use the smallest command that proves the change, then broaden when the change
touches shared behavior:

```bash
pytest tests/unit/test_query_pipeline.py
pytest tests/unit/
pytest tests/integration/
pytest -m "not slow"
pytest
```

Relevant markers include `tier1` through `tier4`, `slow`, `integration`, `e2e`,
`e2e_parser`, `e2e_krag`, `sqlite`, `llm`, `performance`, `scalability`,
`security`, `chaos`, and `property`.

Benchmarks are not substitutes for unit tests. Preserve frozen holdouts and do
not tune implementation details against evaluation-only query labels. Benchmark
methodology lives in [benchmarks/README.md](benchmarks/README.md).

## Style

- Black, isort, and ruff use line length 100.
- Public APIs need type hints and useful docstrings.
- Prefer protocols and existing package boundaries over parallel abstractions.
- Keep source files ASCII unless the existing file and content require Unicode.
- Do not include unrelated refactors or generated artifacts in a change.

```bash
black fitz_sage tests
isort fitz_sage tests
ruff check fitz_sage tests
```

## Pull Requests

Describe:

- the behavior changed and its owner;
- why the previous behavior was wrong or insufficient;
- tests and benchmarks run;
- public contract or limitation changes;
- any remaining risk.

Before requesting review:

- [ ] focused tests pass;
- [ ] architecture contract passes;
- [ ] no compatibility shim or dead alternate path was added;
- [ ] source cleanup and governance remain at their documented owners;
- [ ] current docs and examples match executable APIs;
- [ ] frozen evaluation data was not used as a tuning target.
