# Fitz Examples

Practical examples for the current SDK.

## Quick Start

```bash
pip install fitz-sage

python examples/01_quickstart.py
```

## Examples

| File | Description | Key Features |
|------|-------------|--------------|
| [`01_quickstart.py`](01_quickstart.py) | Basic SDK usage | `fitz()` -> `evidence(..., source=...)` |
| [`02_tabular_sql.py`](02_tabular_sql.py) | CSV -> SQL queries | Native SQLite tables, optional computed answers |
| [`03_local_ollama.py`](03_local_ollama.py) | Optional local synthesis | Ollama OpenAI-compatible endpoint + SQLite |
| [`05_advanced_queries.py`](05_advanced_queries.py) | Query intelligence | Keyword matching, comparisons, aggregations |

## Basic SDK Usage

```python
from fitz_sage import fitz

f = fitz(collection="my_docs")

# The first evidence call can point at a source. Fitz registers the files and
# retrieves governed evidence through the same call.
pack = f.evidence("What is the refund policy?", source="./docs")

print(pack.mode)
for item in pack.items:
    print(f"  - {item.file_path}: {item.excerpt}")
```

Subsequent queries can omit `source` and use the same collection:

```python
pack = f.evidence("What are the cancellation terms?")
```

## Tabular Data

```python
from fitz_sage import fitz

f = fitz(collection="sales")

# Retrieval-first evidence over table metadata and nearby documents
pack = f.evidence("What is the total revenue by region?", source="./data")
```

## Optional Local Synthesis

```bash
ollama pull llama3.2
ollama serve
```

```python
from fitz_sage import fitz

f = fitz(collection="private_docs")
answer = f.query("Summarize the key points", source="./confidential")
```

Point `~/.fitz/config/fitz_krag.yaml` at Ollama's OpenAI-compatible endpoint
when you want generated answers:

```yaml
synthesizer: endpoint/llama3.2
chat_base_url: http://localhost:11434/v1
```

## Advanced Query Features

```python
from fitz_sage import fitz

f = fitz(collection="bugs")
f.evidence("What is BUG-1001?", source="./bug_reports")

pack = f.evidence("Compare Pro vs Enterprise plan")
pack = f.evidence("What are the main trends this quarter?")
pack = f.evidence("What is BUG-9999?")  # should abstain when evidence is missing
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
