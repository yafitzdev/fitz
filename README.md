# fitz-ai ✨

**RAG in 5 minutes. No infrastructure. No boilerplate.**

```bash
pip install fitz-ai

fitz quickstart ./docs "What is our refund policy?"
```

That's it. Your documents are now searchable with AI.

---

## The Problem with RAG Today ☔️

Building RAG shouldn't require a PhD in prompt engineering. Yet here we are:

| Framework | Lines to "Hello World" | External Services | Time to First Answer |
|-----------|------------------------|-------------------|---------------------|
| LangChain | 50+ | Vector DB required | 30+ min |
| LlamaIndex | 30+ | Vector DB required | 20+ min |
| **fitz-ai** | **2** | **None** | **5 min** |

**LangChain** gives you infinite flexibility—and infinite ways to shoot yourself in the foot. Chains, agents, callbacks, memory modules... great for building the next ChatGPT, overkill for "search my docs."

**LlamaIndex** is better for pure retrieval, but still assumes you want to configure chunk sizes, embedding models, index types, and vector stores before asking your first question.

**fitz-ai** assumes you want answers. Configuration is optional.

---

## What Makes Fitz Different ☀️

### 1. Zero-Config Start, Full Control Later

```bash
# Day 1: Just works
fitz quickstart ./contracts "What are the payment terms?"

# Day 30: Full customization when you need it
fitz init                    # Configure everything
fitz ingest ./docs           # Fine-tune chunking
fitz query "..." --retrieval dense_rerank
```

Most RAG frameworks force you to understand the entire stack before you can ask a single question. Fitz inverts this: start with answers, learn the internals only if you need them.

### 2. Epistemic Honesty Built-In

When your documents don't contain the answer, fitz says so:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents. 
    The available financial data covers Q1-Q3 only."
    
    Mode: ABSTAIN
```

Three constraint plugins run automatically:
- **ConflictAwareConstraint**: Detects contradictions across sources
- **InsufficientEvidenceConstraint**: Blocks confident answers without evidence  
- **CausalAttributionConstraint**: Prevents hallucinated cause-effect claims

No prompt engineering required. No "be careful" system messages. Just honest answers.

### 3. YAML-Defined Everything

```yaml
# Add a new LLM provider: just drop a YAML file
# fitz_ai/llm/chat/my_provider.yaml
plugin_name: "my_provider"
plugin_type: "chat"
endpoint:
  path: "/v1/chat/completions"
  method: "POST"
defaults:
  model: "my-model"
  temperature: 0.2
```

LangChain requires you to subclass `BaseChatModel` and implement 47 methods. Fitz plugins are YAML files. Adding OpenAI, Cohere, Anthropic, Ollama, or your custom endpoint is the same: describe the API, done.

---

## Quick Start 🚀

```bash
pip install fitz-ai

fitz quickstart ./docs "Your question here"
```

That's it. Fitz will prompt you for anything it needs.

Want to go fully local with Ollama? No problem:

```bash
pip install fitz-ai

ollama pull llama3.2
ollama pull nomic-embed-text

fitz quickstart ./docs "Your question here"
```

No data leaves your machine. No API costs. Same interface.

**Data privacy**: Fitz runs entirely on your infrastructure. No telemetry, no cloud, no external calls except to the LLM provider you configure.

**Production deployments**: For Qdrant, Docker, and API serving, see the [Deployment Guide](docs/deployment.md).

---

## Real-World Usage 💼

### Chatbot Backend

Fitz handles the hard part—retrieval and grounded answers. Add your conversation layer on top.

```python
from fitz_ai import run

def handle_message(user_message: str) -> str:
    answer = run(user_message, collection="support_docs")
    return answer.text

# Plug into Slack, Discord, Teams, or your own UI
```

No hallucinations. Sources included. You build the chat UX, fitz builds the knowledge.

### Company Knowledge Base

```bash
# Ingest your company docs once
fitz ingest ./wiki ./policies ./runbooks

# Anyone can query
fitz query "How do I request PTO?"
fitz query "What's the incident response process?"
fitz query "Who approves expenses over $5000?"
```

### Customer Support Automation

```bash
fitz ingest ./help-center ./faqs ./product-docs --collection support

fitz query "How do I reset my password?"
fitz query "What's the refund policy?"
fitz query "Why is my payment failing?"
```

Plug the answers into your support chatbot, help desk, or internal tools.

### Research & Literature Review

```bash
fitz ingest ./papers --collection research

fitz query "What methods achieve SOTA on ImageNet?"
fitz query "Compare transformer vs CNN approaches"
fitz query "Which papers cite attention mechanisms for NLP?"
```

---

## Architecture 🏛

```
┌─────────────────────────────────────────────────────────────┐
│                         fitz-ai                             │
├─────────────────────────────────────────────────────────────┤
│  CLI Layer                                                  │
│  quickstart | init | ingest | query | config | doctor       │
├─────────────────────────────────────────────────────────────┤
│  Engines                                                    │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │ Classic RAG │  │   CLaRa     │  (pluggable)              │
│  └─────────────┘  └─────────────┘                           │
├─────────────────────────────────────────────────────────────┤
│  Plugin System (all YAML-defined)                           │
│  ┌────────┐ ┌───────────┐ ┌────────┐ ┌─────────┐            │
│  │  LLM   │ │ Embedding │ │ Rerank │ │VectorDB │            │
│  └────────┘ └───────────┘ └────────┘ └─────────┘            │
│  openai, cohere, anthropic, ollama, azure...                │
├─────────────────────────────────────────────────────────────┤
│  Retrieval Pipelines (YAML-composed)                        │
│  dense.yaml | dense_rerank.yaml | custom...                 │
├─────────────────────────────────────────────────────────────┤
│  Constraints (epistemic safety)                             │
│  ConflictAware | InsufficientEvidence | CausalAttribution   │
└─────────────────────────────────────────────────────────────┘
```

---

## CLI Reference

```bash
fitz quickstart [PATH] [QUESTION]  # Zero-config RAG (start here)
fitz init                          # Interactive setup wizard
fitz ingest [PATH]                 # Ingest documents
fitz query "question"              # Query knowledge base
fitz config                        # View/edit configuration
fitz doctor                        # System diagnostics
```

---

## Comparison ⚖️

| | fitz-ai | LangChain | LlamaIndex |
|--|---------|-----------|------------|
| Time to first answer | 5 min | 30+ min | 20+ min |
| Config required to start | None | Yes | Yes |
| Knows when to say "I don't know" | Built-in | DIY | DIY |
| Add new LLM provider | Drop a YAML | Subclass + 200 LOC | Subclass + 150 LOC |
| Swap retrieval paradigm | Change 1 line | Rewrite pipeline | Rewrite pipeline |

**Choose fitz-ai if**: You want to query your documents with AI, not build an AI platform.

**Choose LangChain if**: You're building complex agent workflows with tool use, memory, and multi-step reasoning.

**Choose LlamaIndex if**: You need deep customization of retrieval strategies across heterogeneous data sources.

---

## Beyond RAG 🔮

> **RAG is a method. Knowledge access is a strategy.**

Fitz is not a RAG framework. It's a knowledge platform that *currently* uses RAG as its primary engine.

```python
from fitz_ai import run

# Today: Classic RAG
answer = run("What are the payment terms?", engine="classic_rag")

# Also available: CLaRa (compressed RAG, 16x smaller context)
answer = run("What are the payment terms?", engine="clara")

# Tomorrow: GraphRAG, HyDE, or whatever comes next
answer = run("What are the payment terms?", engine="graph_rag")
```

The engine is an implementation detail. Your ingested knowledge, your queries, your workflow—all stay the same. When a better retrieval paradigm emerges, swap one line, not your entire codebase.

---

## Philosophy 📍

**Principles:**
- **Explicit over clever**: No magic. Read the config, know what happens.
- **Answers over architecture**: Optimize for time-to-insight, not flexibility.
- **Honest over helpful**: Better to say "I don't know" than hallucinate.
- **Files over frameworks**: YAML plugins over class hierarchies.

---

## License

MIT

---

## About

Solo project by [Yan Fitzner](https://github.com/yafitzdev). ~15k lines of Python. 400+ tests. Built from scratch—no LangChain or LlamaIndex under the hood.

---

## Links

- [GitHub](https://github.com/yafitzdev/fitz-ai)
- [PyPI](https://pypi.org/project/fitz-ai/)
- [Changelog](CHANGELOG.md)
- [CLI Documentation](docs/CLI.md)