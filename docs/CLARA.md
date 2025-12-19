# CLaRa Engine

CLaRa (Continuous Latent Reasoning) is Apple's compression-native RAG paradigm that compresses documents into continuous memory tokens while preserving semantic content.

## Overview

Unlike traditional RAG which retrieves text chunks, CLaRa:
- Compresses documents 16x-128x into continuous embeddings
- Performs retrieval in latent space via cosine similarity
- Generates answers directly from compressed representations
- Uses a single model for both retrieval and generation

## Hardware Requirements

> ⚠️ **Important**: CLaRa requires significant compute resources.

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU VRAM** | 12GB | 16GB+ |
| **System RAM** | 16GB | 32GB+ |
| **Disk Space** | 14GB | 20GB |
| **GPU** | RTX 3080 | RTX 4090 / A100 |

**Without adequate hardware**, the engine will fail to load the 7B parameter model.

For development and testing without a powerful GPU:
- Use the **Classic RAG engine** instead
- Run unit tests (they use mocked models)
- Wait for smaller CLaRa variants (MLX support planned)

## Installation

```bash
# Install with CLaRa dependencies
pip install fitz[clara]

# Or manually
pip install transformers torch
```

## Quick Start

```python
from fitz_ai.engines.clara import create_clara_engine
from fitz_ai.core import Query

# Create engine (downloads ~14GB model on first run)
engine = create_clara_engine()

# Add documents
engine.add_documents([
    "Quantum computing uses qubits instead of classical bits...",
    "Machine learning enables computers to learn from data...",
    "Neural networks are inspired by biological neurons...",
])

# Query
answer = engine.answer(Query(text="What is quantum computing?"))
print(answer.text)
```

## Configuration

### Default Configuration

```python
from fitz_ai.engines.clara.config.schema import ClaraConfig

config = ClaraConfig()
# Uses apple/CLaRa-7B-Instruct with 16x compression
```

### Custom Configuration

```python
from fitz_ai.engines.clara.config.schema import (
    ClaraConfig,
    ClaraModelConfig,
    ClaraCompressionConfig,
    ClaraRetrievalConfig,
    ClaraGenerationConfig,
)

config = ClaraConfig(
    model=ClaraModelConfig(
        model_name_or_path="apple/CLaRa-7B-E2E",
        variant="e2e",
        device="cuda",
        torch_dtype="bfloat16",
        load_in_8bit=True,  # Reduce VRAM usage
    ),
    compression=ClaraCompressionConfig(
        compression_rate=16,  # 16x or 128x
        doc_max_length=2048,
    ),
    retrieval=ClaraRetrievalConfig(
        top_k=5,
    ),
    generation=ClaraGenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
    ),
)

engine = create_clara_engine(config=config)
```

### YAML Configuration

```yaml
# clara_config.yaml
clara:
  model:
    model_name_or_path: "apple/CLaRa-7B-Instruct"
    variant: "instruct"
    device: "cuda"
    torch_dtype: "bfloat16"
  
  compression:
    compression_rate: 16
    doc_max_length: 2048
  
  retrieval:
    top_k: 5
  
  generation:
    max_new_tokens: 256
    temperature: 0.7
```

## Model Variants

| Variant | Model ID | Use Case |
|---------|----------|----------|
| **Base** | `apple/CLaRa-7B-Base` | Document compression, paraphrase generation |
| **Instruct** | `apple/CLaRa-7B-Instruct` | Instruction-following QA (recommended) |
| **E2E** | `apple/CLaRa-7B-E2E` | End-to-end retrieval + generation |

Each variant is available with two compression rates:
- `compression-16` (16x compression)
- `compression-128` (128x compression)

## API Reference

### ClaraEngine

```python
class ClaraEngine:
    def __init__(self, config: ClaraConfig):
        """Initialize with configuration."""
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to knowledge base. Returns document IDs."""
    
    def answer(self, query: Query) -> Answer:
        """Execute query and return answer with provenance."""
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
```

### Convenience Functions

```python
from fitz_ai.engines.clara import (
    run_clara,           # Quick one-off query
    create_clara_engine, # Create reusable engine
    clear_engine_cache,  # Clear cached engine
)
```

## Comparison with Classic RAG

| Feature | Classic RAG | CLaRa |
|---------|-------------|-------|
| **Storage** | Vector embeddings | Compressed memory tokens |
| **Compression** | None | 16x-128x |
| **Retrieval** | Separate embedding model | Built into LLM |
| **Training** | Separate retriever/generator | End-to-end unified |
| **Multi-hop** | Limited | Superior |
| **Hardware** | Moderate | High |
| **Production Ready** | ✅ Yes | ⚠️ Experimental |

## When to Use CLaRa

✅ **Good fit:**
- Research and experimentation
- Large document collections
- Complex multi-hop reasoning
- When you have GPU resources

❌ **Not recommended:**
- Production deployments (use Classic RAG)
- Limited hardware
- Simple single-hop queries
- Real-time/low-latency requirements

## Troubleshooting

### Out of Memory (OOM)

```python
# Try 8-bit quantization
config = ClaraConfig(
    model=ClaraModelConfig(
        load_in_8bit=True,
    )
)
```

### Model Not Found

The model path must match HuggingFace exactly. Use `subfolder` for compression variants:

```python
# Correct: Use the HuggingFace model ID
model_name_or_path="apple/CLaRa-7B-Instruct"

# The compression rate is specified separately
compression_rate=16
```

### Slow First Run

The first run downloads the model (~14GB). This is normal and only happens once.

## References

- [CLaRa Paper](https://arxiv.org/abs/2511.18659)
- [Apple ML-CLaRa GitHub](https://github.com/apple/ml-clara)
- [HuggingFace Models](https://huggingface.co/apple)