# Fitz Examples

Practical examples for the current SDK.

## Quick Start

```bash
pip install fitz-sage
export OPENAI_API_KEY="your-key"  # if using a hosted OpenAI-compatible endpoint

python examples/01_quickstart.py
```

## Examples

| File | Description | Key Features |
|------|-------------|--------------|
| [`01_quickstart.py`](01_quickstart.py) | Basic SDK usage | `fitz()` -> `query(..., source=...)` |
| [`02_tabular_sql.py`](02_tabular_sql.py) | CSV -> SQL queries | Native SQLite tables, computed answers |
| [`03_local_ollama.py`](03_local_ollama.py) | 100% local setup | No API keys, Ollama + SQLite |
| [`05_advanced_queries.py`](05_advanced_queries.py) | Query intelligence | Keyword matching, comparisons, aggregations |

## Basic SDK Usage

```python
from fitz_sage import fitz

f = fitz(collection="my_docs")

# The first query can point at a source. Fitz registers the files and queries
# the collection through the same call.
answer = f.query("What is the refund policy?", source="./docs")

print(answer.text)
for source in answer.provenance:
    print(f"  - {source.source_id}")
```

Subsequent queries can omit `source` and use the same collection:

```python
answer = f.query("What are the cancellation terms?")
```

## Tabular Data

```python
from fitz_sage import fitz

f = fitz(collection="sales")

# Natural language -> SQL -> computed answer
answer = f.query("What is the total revenue by region?", source="./data")
```

## Local-Only

```bash
ollama pull llama3.2
ollama serve
```

```python
from fitz_sage import fitz

f = fitz(collection="private_docs")
answer = f.query("Summarize the key points", source="./confidential")
```

Point `~/.fitz/config/fitz_krag.yaml` at Ollama's OpenAI-compatible endpoint:

```yaml
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: http://localhost:11434/v1
chat_smart_model: llama3.2
```

## Advanced Query Features

```python
from fitz_sage import fitz

f = fitz(collection="bugs")
f.query("What is BUG-1001?", source="./bug_reports")

answer = f.query("Compare Pro vs Enterprise plan")
answer = f.query("What are the main trends this quarter?")
answer = f.query("What is BUG-9999?")  # should abstain when evidence is missing
```

## Running the Examples

Each example is self-contained except `01_quickstart.py`, which expects a
local `./docs` folder unless you change the `source` variable.

```bash
python examples/01_quickstart.py
python examples/02_tabular_sql.py
python examples/03_local_ollama.py
python examples/05_advanced_queries.py
```

## CLI Quick Reference

```bash
fitz query "Your question" --source ./docs
fitz collections
fitz serve
```

## More Resources

- [Full Documentation](../docs/)
- [Configuration Guide](../docs/CONFIG.md)
- [CLI Reference](../docs/CLI.md)
- [Architecture](../docs/ARCHITECTURE.md)
