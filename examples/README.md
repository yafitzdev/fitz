# fitz Examples

This directory contains example scripts demonstrating fitz usage.

## Examples

| File | Description |
|------|-------------|
| `quickstart.py` | Basic RAG pipeline flow without external dependencies |
| `ingestion_example.py` | Document ingestion and chunking example |
| `full_pipeline.py` | End-to-end RAG with vector DB (requires Qdrant) |

## Running Examples

### Quickstart (No External Dependencies)

```bash
python examples/quickstart.py
```

This demonstrates:
- Context pipeline processing
- RGS prompt building
- Answer structuring with citations

### Ingestion Example

```bash
# Create some test documents first
mkdir -p test_docs
echo "This is a test document about RAG systems." > test_docs/doc1.txt
echo "Vector databases store embeddings for similarity search." > test_docs/doc2.txt

# Run the example
python examples/ingestion_example.py
```

### Full Pipeline (Requires Setup)

1. Start Qdrant:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. Set API keys:
   ```bash
   export COHERE_API_KEY="your-key"
   ```

3. Run:
   ```bash
   python examples/full_pipeline.py
   ```

## Creating Your Own Examples

Use the examples as templates. Key patterns:

```python
# Config-driven pipeline creation
from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import create_pipeline_from_yaml

pipeline = create_pipeline_from_yaml("my_config.yaml")

# Direct component usage
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSConfig

rgs = RGS(RGSConfig(enable_citations=True))

# Plugin selection via registry
from fitz_ai.llm import get_llm_plugin

ChatPlugin = get_llm_plugin(plugin_name="cohere", plugin_type="chat")
```
