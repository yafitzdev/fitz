# tests/README.md
# fitz-sage Test Suite

The test tree is split by risk and runtime cost. Unit tests are the default
developer loop; E2E tests exercise KRAG ingestion, retrieval, reranking, and
Pyrrho governance over fixture corpora.

## Test Categories

| Category | Purpose | Marker | Run Command |
|---|---|---|---|
| Unit | Fast component tests | - | `pytest tests/unit/` |
| KRAG E2E | End-to-end retrieval behavior | `e2e_krag` | `pytest tests/e2e_krag/` |
| Parser E2E | PDF/DOCX/parser coverage | `e2e_krag_parser` | `pytest -m e2e_krag_parser` |
| Format E2E | PPTX/XLSX/SQL/Go/Java/TS fixtures | `e2e_krag_formats` | `pytest -m e2e_krag_formats` |
| Integration | Real-service or cross-layer tests | `integration` | `pytest -m integration` |
| Performance | Latency and throughput benchmarks | `performance` | `pytest -m performance` |
| Scalability | Large corpus and concurrent load | `scalability` | `pytest -m scalability` |
| Security | Injection, leakage, validation | `security` | `pytest -m security` |
| Chaos | Reliability and failure handling | `chaos` | `pytest -m chaos` |

## Quick Start

```bash
# Fast local loop
pytest tests/unit/ -v

# Unit tests plus KRAG E2E
pytest tests/unit/ tests/e2e_krag/ -v

# Everything except slow/scalability work
pytest -m "not slow and not scalability"

# Coverage
pytest --cov=fitz_sage --cov-report=html
```

## Test Structure

```text
tests/
├── test_config.yaml          # Shared endpoint config for tests that need chat
├── conftest.py               # Root fixtures
├── unit/                     # Fast isolated tests
│   ├── llm/                  # Auth/provider/factory tests
│   ├── property/             # Hypothesis strategies and property tests
│   └── tabular/              # CSV/table parser/query/store tests
├── e2e_krag/                 # KRAG end-to-end fixtures and scenarios
│   ├── fixtures_formats/     # PPTX/XLSX/SQL/Go/Java/TypeScript samples
│   ├── fixtures_parser/      # PDF/DOCX parser samples
│   ├── fixtures_rag/         # Retrieval/governance fixture corpus
│   ├── scenarios.py          # E2E scenario definitions
│   └── runner.py             # E2E runner
├── integration/              # Cross-layer integration tests
├── performance/              # Latency benchmarks
├── load/                     # Locust and scalability tests
├── security/                 # Prompt injection, leakage, input validation
└── chaos/                    # Failure-mode tests
```

## E2E Coverage Areas

The KRAG E2E suite covers:

- exact identifier and sparse FTS5 retrieval;
- code symbol retrieval and import graph expansion;
- table schema retrieval and SQL-backed table querying;
- comparison, temporal, aggregation, and multi-query behavior;
- entity graph expansion and multi-hop retrieval;
- insufficient-evidence and disputed-evidence governance;
- PDF/DOCX/parser and mixed-format retrieval fixtures.

## Running Load Tests

```bash
cd tests/load
locust -f locustfile.py --headless -u 10 -r 2 -t 60s
```

## Adding New Tests

1. Unit behavior: add `tests/unit/test_<feature>.py`.
2. KRAG E2E scenario: add to `tests/e2e_krag/scenarios.py` and cover it in the runner/tests.
3. Parser or format fixture: add under `tests/e2e_krag/fixtures_parser/` or `fixtures_formats/`.
4. Security/performance/chaos: add to the matching top-level test directory.
