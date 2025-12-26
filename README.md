# fitz-ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-ai.svg)](https://pypi.org/project/fitz-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.1-green.svg)](CHANGELOG.md)

**Honest RAG in 5 minutes. Says "I don't know" when it should. No infrastructure. No boilerplate.**

```bash
pip install fitz-ai

fitz quickstart ./docs "What is our refund policy?"
```

That's it. Your documents are now searchable with AI.

![fitz-ai quickstart demo](docs/assets/quickstart_demo.gif)

---

## About

Solo project by [Yan Fitzner](https://github.com/yafitzdev). ~40k lines of Python. 400+ tests. 

Built from scratch—no LangChain or LlamaIndex under the hood.


---

## Why Fitz? 💡

1. [x] **Point at a folder. Ask a question. Get an answer with sources.**
2. [x] **❗Swap RAG engines in one line, no migration needed.** Fitz is designed for people who anticipate RAG to evolve.
3. [x] **Admits it doesn't know when the answer isn't there.** No hallucinations, no confident nonsense.
4. [x] **Local execution possible.** FAISS and Ollama support, no API keys required to start.
5. [x] **Plugin-based architecture.** Swap LLMs, vector databases, rerankers, and retrieval pipelines via YAML config.
6. [x] **Full provenance.** Every answer traces back to the exact chunk and document.
7. [x] **Data privacy**: No telemetry, no cloud, no external calls except to the LLM provider you configure.

Any questions left? Try fitz on itself:

```bash
fitz quickstart ./fitz_ai "How does the chunking pipeline work?"
```

The codebase speaks for itself.

---
<details>

<summary><strong>📦 Fitz vs LangChain vs LlamaIndex</strong></summary>

#### Fitz opts for a deliberately narrower approach.

LangChain and LlamaIndex are powerful **LLM application frameworks** designed to help developers build complex, end-to-end AI systems. 
Fitz provides a **minimal, replaceable RAG engine** with strong epistemic guarantees — without locking users into a framework, ecosystem, or long-term architectural commitment.

Fitz is not a competitor in scope.  
It is an infrastructure primitive.

#### Core philosophical differences ⚖️

| Dimension | Fitz | LangChain | LlamaIndex |
|--------|------|-----------|------------|
| Primary role | **RAG engine** | LLM application framework | LLM data framework |
| User commitment | **No framework lock-in** | High | High |
| Engine coupling | **Swappable in one line** | Deep | Deep |
| Design goal | Correctness & honesty | Flexibility | Data integration |
| Long-term risk | Low | Migration-heavy | Migration-heavy |


#### Epistemic behavior (truth over fluency) 🎯

| Aspect | Fitz | LangChain / LlamaIndex |
|-----|------|------------------------|
| “I don’t know” | **First-class behavior** | Not guaranteed |
| Hallucination handling | Designed-in | Usually prompt-level |
| Confidence signaling | Explicit | Implicit |

Fitz treats uncertainty as a **feature**, not a failure.  
If the system cannot support an answer with retrieved evidence, it says so.

#### Transparency & provenance 🔎

| Capability | Fitz | LangChain / LlamaIndex |
|---------|------|------------------------|
| Source attribution | **Mandatory** | Optional |
| Retrieval trace | **Explicit & structured** | Often opaque |
| Debuggability | Built-in | Tool-dependent |

Every answer in Fitz is fully auditable down to the retrieval step.

#### Scope & complexity 🪐

| Aspect | Fitz | LangChain / LlamaIndex |
|-----|------|------------------------|
| Chains / agents | ❎ | ✔ |
| Prompt graphs | ❎ | ✔ |
| UI abstractions | ❎ | Often |
| Cognitive overhead | **Very low** | High |

Fitz intentionally does less — so it can be trusted more.
</details>

---

<details>

<summary><strong>📦 When Fitz is the right choice</strong></summary>

#### Use Fitz if you want:

- A replaceable RAG engine, not a framework marriage
- Strong epistemic guarantees (“I don’t know” is valid output)
- Full provenance for every answer
- A transparent, extensible plugin architecture
- A future-proof ingestion pipeline that survives engine changes

</details>

---
<details>

<summary><strong>📦 Features</strong></summary>

#### Actually admits when it doesn't know 📚

When documents don't contain the answer, fitz says so:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents.
    The available financial data covers Q1-Q3 only."

    Mode: ABSTAIN
```

Three constraint plugins run automatically:
- **📕 ConflictAwareConstraint**: Detects contradictions across sources
- **📗 InsufficientEvidenceConstraint**: Blocks answers without evidence
- **📘 CausalAttributionConstraint**: Prevents hallucinated cause-effect claims

#### Full Provenance 🗂️

Every answer traces back to its source:

```
Answer: The refund policy allows returns within 30 days...

Sources:
  [1] policies/refund.md [chunk 3] (score: 0.92)
  [2] faq/payments.md [chunk 1] (score: 0.87)
```

#### Swappable RAG Engines 🔄

Your data stays. Your queries stay. Only the engine changes.

```
        ┌─────────────────────────────────────┐
        │           Your Query                │
        │   "What are the payment terms?"     │
        └──────────────────┬──────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │       engine="..."                  │
        │  ┌─────────┐ ┌───────┐ ┌─────────┐  │
        │  │ classic │ │ clara │ │ graph   │  │
        │  │  _rag   │ │       │ │  _rag   │  │
        │  └────┬────┘ └───┬───┘ └────┬────┘  │
        │       └──────────┼──────────┘       │
        └──────────────────┼──────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │       Your Ingested Knowledge       │
        │      (unchanged across engines)     │
        └─────────────────────────────────────┘
```

```python
answer = run("What are the payment terms?", engine="classic_rag")
answer = run("What are the payment terms?", engine="clara")
answer = run("What are the payment terms?", engine="graph_rag")  # future
```

No migration. No re-ingestion. No new API to learn.

#### Enrichment 📥

Opt-in enrichment plugins enhance your knowledge base:

- **Code-derived artifacts**: Navigation indexes, interface catalogs, dependency graphs—extracted directly from your codebase via AST analysis. No LLM required.
- **LLM-generated summaries**: Natural language descriptions for chunks, making code more discoverable via semantic search.

Your question matches enriched context, not just raw text. Fully extensible—add your own enrichment plugins.
</details>

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

---

## Real-World Usage

Fitz is a foundation. It handles document ingestion and grounded retrieval—you build whatever sits on top: chatbots, dashboards, alerts, or automation.

<strong>Chatbot Backend 🤖</strong>

> Connect fitz to Slack, Discord, Teams, or your own UI. One function call returns an answer with sources—no hallucinations, full provenance. You handle the conversation flow; fitz handles the knowledge.
>
> *Example:* A SaaS company plugs fitz into their support bot. Tier-1 questions like "How do I reset my password?" get instant answers. Their support team focuses on edge cases while fitz deflects 60% of incoming tickets.

<strong>Internal Knowledge Base 📖</strong>

> Point fitz at your company's wiki, policies, and runbooks. Employees ask natural language questions instead of hunting through folders or pinging colleagues on Slack.
>
> *Example:* A 200-person startup ingests their Notion workspace and compliance docs. New hires find answers to "How do I request PTO?" on day one—no more waiting for someone in HR to respond.

<strong>Continuous Intelligence & Alerting (Watchdog) 🐶</strong>

> Pair fitz with cron, Airflow, or Lambda. Ingest data on a schedule, run queries automatically, trigger alerts when conditions match. Fitz provides the retrieval primitive; you wire the automation.
>
> *Example:* A security team ingests SIEM logs nightly. Every morning, a scheduled job asks "Were there failed logins from unusual locations?" If fitz finds evidence, an alert fires to the on-call channel before anyone checks email.

<strong>Web Knowledge Base 🌎</strong>

> Scrape the web with Scrapy, BeautifulSoup, or Playwright. Save to disk, ingest with fitz. The web becomes a queryable knowledge base.
>
> *Example:* A football analytics hobbyist scrapes Premier League match reports. After ingesting, they ask "How did Arsenal perform against top 6 teams?" or "What tactics did Liverpool use in away games?"—insights that would take hours to compile manually.

<strong>Codebase Search 🐍</strong>

> Fitz includes built-in AST-aware chunking for Python. Functions, classes, and modules become individual searchable units with docstrings and imports preserved. Ask questions in natural language; get answers pointing to specific code.
>
> *Example:* A team inherits a legacy Django monolith—200k lines, sparse docs. They ingest the codebase and ask "Where is user authentication handled?" or "What API endpoints modify the billing table?" New developers onboard in days instead of weeks.

---

## Architecture 🏛

```
┌───────────────────────────────────────────────────────────────┐
│                         fitz-ai                               │
├───────────────────────────────────────────────────────────────┤
│  CLI Layer                                                    │
│  quickstart | init | ingest | query | chat | config | doctor  │
├───────────────────────────────────────────────────────────────┤
│  Engines                                                      │
│  ┌───────────────┐  ┌───────────┐                             │
│  │  Classic RAG  │  │   CLaRa   │  (pluggable)                │
│  └───────────────┘  └───────────┘                             │
├───────────────────────────────────────────────────────────────┤
│  Plugin System (all YAML-defined)                             │
│  ┌────────┐ ┌───────────┐ ┌────────┐ ┌──────────┐             │
│  │  LLM   │ │ Embedding │ │ Rerank │ │ VectorDB │             │
│  └────────┘ └───────────┘ └────────┘ └──────────┘             │
│  openai, cohere, anthropic, ollama, azure...                  │
├───────────────────────────────────────────────────────────────┤
│  Retrieval Pipelines (YAML-composed)                          │
│  dense.yaml | dense_rerank.yaml | custom...                   │
├───────────────────────────────────────────────────────────────┤
│  Enrichment (opt-in)                                          │
│  code artifacts | LLM summaries | custom plugins              │
├───────────────────────────────────────────────────────────────┤
│  Constraints (epistemic safety)                               │
│  ConflictAware | InsufficientEvidence | CausalAttribution     │
└───────────────────────────────────────────────────────────────┘
```
---

## CLI Reference ⌨️

```bash
fitz quickstart [PATH] [QUESTION]    # Zero-config RAG (start here)
fitz init                            # Interactive setup wizard
fitz ingest                          # Interactive ingestion
fitz query                           # Single question with sources
fitz chat                            # Multi-turn conversation with your knowledge base
fitz collections                     # List and delete knowledge collections
fitz config                          # View/edit configuration
fitz doctor                          # System diagnostics
```

---

## Beyond RAG 🔭

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

## Philosophy 🧭

**Principles:**
- **Explicit over clever**: No magic. Read the config, know what happens.
- **Answers over architecture**: Optimize for time-to-insight, not flexibility.
- **Honest over helpful**: Better to say "I don't know" than hallucinate.
- **Files over frameworks**: YAML plugins over class hierarchies.

---

## License 🪪

MIT

---

## Links 🔗

- [GitHub](https://github.com/yafitzdev/fitz-ai)
- [PyPI](https://pypi.org/project/fitz-ai/)
- [Changelog](CHANGELOG.md)
- [CLI Documentation](docs/CLI.md)