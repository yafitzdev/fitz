# Fitz AI - CLI Documentation

## Overview

Fitz provides a clean, minimal command-line interface for local-first RAG (Retrieval-Augmented Generation).

---

## Core Commands

### Setup & Configuration

#### `fitz init`
Interactive setup wizard that detects your system and creates a working configuration.

**Usage:**
```bash
fitz init              # Interactive wizard
fitz init -y           # Auto-detect and use defaults
fitz init --show       # Preview config without saving
```

**What it does:**
- Detects available providers (Ollama, Qdrant, API keys)
- Discovers available plugins
- Prompts for configuration choices
- Generates and saves `fitz.yaml` config file

**Examples:**
```bash
# Interactive setup
fitz init

# Non-interactive with defaults
fitz init -y

# Preview without saving
fitz init --show
```

---

#### `fitz config`
View and manage your Fitz configuration.

**Usage:**
```bash
fitz config              # Show config summary
fitz config --raw        # Show raw YAML
fitz config --json       # Output as JSON
fitz config --path       # Show config file path
fitz config --edit       # Open in editor
```

**Examples:**
```bash
# View configuration summary
fitz config

# See the raw YAML file
fitz config --raw

# Get JSON output (for scripting)
fitz config --json

# Edit config in your $EDITOR
fitz config --edit

# Get path to config file
fitz config --path
```

---

### Ingestion

#### `fitz ingest`
Ingest documents into your vector database.

**Usage:**
```bash
fitz ingest [SOURCE]              # Interactive mode
fitz ingest [SOURCE] -y           # Non-interactive with defaults
```

**Options:**
- `SOURCE` - Path to file or directory (optional, will prompt if not provided)
- `-y, --non-interactive` - Use defaults without prompting

**What it does:**
1. Reads documents from source path
2. Chunks documents using configured chunker
3. Generates embeddings
4. Stores vectors in configured vector database

**Examples:**
```bash
# Interactive ingestion (prompts for all options)
fitz ingest

# Ingest specific directory interactively
fitz ingest ./documents

# Non-interactive with defaults
fitz ingest ./documents -y

# The interactive mode will prompt for:
#   - Source path (if not provided)
#   - Collection name
#   - Chunker type
#   - Chunk size
#   - Chunk overlap
```

**Configuration from CLI prompts:**
When run interactively, you can configure:
- **Collection name**: Where to store the documents (default: from config)
- **Chunker**: Which chunking strategy to use (e.g., "simple")
- **Chunk size**: Size of text chunks in characters (default: 1000)
- **Chunk overlap**: Overlap between chunks (default: 0)

---

### Querying

#### `fitz query`
Query your knowledge base using RAG.

**Usage:**
```bash
fitz query "your question"
fitz query "your question" --stream
```

**Options:**
- `QUESTION` - Your query text (required)
- `--stream` - Stream the response (if supported by LLM)

**What it does:**
1. Embeds your query
2. Retrieves relevant chunks from vector database
3. Processes and ranks context
4. Generates answer using LLM
5. Returns answer with source citations

**Examples:**
```bash
# Basic query
fitz query "What is RAG?"

# Stream the response
fitz query "Explain quantum computing" --stream

# Complex queries with quotes
fitz query "What are the main topics in the ingested documents?"
```

**Output includes:**
- Generated answer text
- Source citations with labels
- Metadata about sources used

---

### Diagnostics

#### `fitz doctor`
Run system diagnostics to verify your setup.

**Usage:**
```bash
fitz doctor              # Quick check
fitz doctor -v           # Verbose output
fitz doctor --test       # Test actual connections
```

**Options:**
- `-v, --verbose` - Show detailed output
- `-t, --test` - Test actual connections to services

**What it checks:**
- Python version (3.10+)
- Workspace and config files
- Required dependencies
- Optional dependencies (in verbose mode)
- Available services (Ollama, Qdrant, FAISS)
- API key configuration
- Connection tests (with --test flag)

**Examples:**
```bash
# Quick health check
fitz doctor

# Detailed check with optional packages
fitz doctor -v

# Full check including connection tests
fitz doctor --test
```

**Sample Output:**
```
Fitz Doctor
===========

System
------
✓ Python       Python 3.11.5
✓ Workspace    /home/user/.fitz
✓ Config       Valid

Dependencies
------------
✓ typer        0.9.0
✓ httpx        0.25.0
✓ pydantic     2.5.0

Services
--------
✓ Ollama       http://localhost:11434
✓ Qdrant       localhost:6333
⚠ FAISS        Not installed

API Keys
--------
✓ Cohere       configured
⚠ OpenAI       $OPENAI_API_KEY not set
```

---

## Quick Start Guide

### 1. Initial Setup
```bash
# Run the setup wizard
fitz init

# Or use auto-detected defaults
fitz init -y
```

### 2. Ingest Documents
```bash
# Ingest your documents
fitz ingest ./my-documents

# Or with specific options
fitz ingest ./my-documents -y
```

### 3. Query Your Knowledge
```bash
# Ask questions
fitz query "What are the main topics?"

# Get streaming responses
fitz query "Explain the architecture" --stream
```

### 4. Verify Setup
```bash
# Check system health
fitz doctor

# Run full diagnostics
fitz doctor --test
```

---

## Configuration File

The `fitz init` command creates a `fitz.yaml` file in `~/.fitz/fitz.yaml` with this structure:

```yaml
# Chat (LLM for answering questions)
chat:
  plugin_name: cohere
  kwargs:
    model: command-a-03-2025
    temperature: 0.2

# Embedding (text to vectors)
embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0

# Vector Database
vector_db:
  plugin_name: qdrant
  kwargs:
    host: "localhost"
    port: 6333

# Retriever
retriever:
  plugin_name: dense
  collection: default
  top_k: 5

# Reranker (optional)
rerank:
  enabled: true
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5

# RGS (Retrieval-Guided Synthesis)
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8

# Logging
logging:
  level: INFO
```

You can:
- Edit manually: `vim ~/.fitz/fitz.yaml`
- Edit via CLI: `fitz config --edit`
- View current: `fitz config`
- View raw: `fitz config --raw`

---

## Environment Variables

Fitz uses environment variables for API authentication:

```bash
# Cohere (recommended for embeddings + chat)
export COHERE_API_KEY="your-key-here"

# OpenAI (alternative)
export OPENAI_API_KEY="your-key-here"

# Azure OpenAI (alternative)
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

---

## Common Workflows

### Local-First Setup (No API Keys)
```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model
ollama pull llama3.2

# 3. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 4. Initialize with local tools
fitz init
# Select: ollama for chat/embedding, qdrant for vector DB

# 5. Ingest and query
fitz ingest ./docs -y
fitz query "What's in my docs?"
```

### Cloud-Based Setup
```bash
# 1. Set API key
export COHERE_API_KEY="your-key"

# 2. Start Qdrant (or use cloud Qdrant)
docker run -p 6333:6333 qdrant/qdrant

# 3. Initialize
fitz init -y

# 4. Use as normal
fitz ingest ./docs -y
fitz query "Your question"
```

### Development Workflow
```bash
# Check your setup is working
fitz doctor --test

# Ingest documents
fitz ingest ./project-docs -y

# Query iteratively
fitz query "How does X work?"
fitz query "What about Y?"

# View configuration
fitz config

# Edit if needed
fitz config --edit
```

---

## Troubleshooting

### "No configuration found"
```bash
# Run init to create config
fitz init
```

### "No chat plugins available"
```bash
# Set an API key OR start Ollama
export COHERE_API_KEY="your-key"
# OR
ollama serve
```

### "Failed to connect to Qdrant"
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# OR install FAISS alternative
pip install faiss-cpu
fitz init  # Select faiss when prompted
```

### "Module not found" errors
```bash
# Install missing dependencies
pip install -e ".[dev]"

# Check what's missing
fitz doctor -v
```

---

## Tips

1. **Use `fitz doctor` first** - Always run diagnostics before reporting issues
2. **Check config path** - Use `fitz config --path` to see where config is stored
3. **Preview before saving** - Use `fitz init --show` to see generated config
4. **Edit directly** - Use `fitz config --edit` to modify config in your editor
5. **Non-interactive mode** - Use `-y` flag for scripting and automation
6. **Streaming responses** - Use `--stream` for real-time query responses

---

## Exit Codes

- `0` - Success
- `1` - Error (check error message for details)

---

## Getting Help

For any command, use `--help`:
```bash
fitz --help
fitz init --help
fitz ingest --help
fitz query --help
fitz config --help
fitz doctor --help
```