# fitz-ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-ai.svg)](https://pypi.org/project/fitz-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.2-green.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/yafitzdev/fitz-ai)

---

**Honest RAG in 5 minutes. No infrastructure. No boilerplate.**

```bash
pip install fitz-ai

fitz quickstart ./docs "What is our refund policy?"
```

That's it. Your documents are now searchable with AI.

![fitz-ai quickstart demo](https://raw.githubusercontent.com/yafitzdev/fitz-ai/main/docs/assets/quickstart_demo.gif)

---

### About 🧑‍🌾

  Solo project by Yan Fitzner ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev)).

  - ~40k lines of Python
  - 600+ tests, 100% coverage
  - Zero LangChain/LlamaIndex dependencies — built from scratch

![fitz-ai honest_rag](https://raw.githubusercontent.com/yafitzdev/fitz-ai/main/docs/assets/honest_rag.jpg)

---

<details>

<summary><strong>📦 What is RAG?</strong></summary>

<br>

RAG is how ChatGPT's "file search," Notion AI, and enterprise knowledge tools actually work under the hood.
Instead of sending all your documents to an AI, RAG:

1. [X] **Indexes your documents once** — Splits them into chunks, converts to vectors, stores in a database
2. [X] **Retrieves only what's relevant** — When you ask a question, finds the 5-10 most relevant chunks
3. [X] **Sends just those chunks to the LLM** — The AI answers based on focused, relevant context

Traditional approach:
```
  [All 10,000 documents] → LLM → Answer
  ❌ Impossible (too large)
  ❌ Expensive (if possible)
  ❌ Unfocused
```
RAG approach:
```
  Question → [Search index] → [5 relevant chunks] → LLM → Answer
  ✅ Works at any scale
  ✅ Costs pennies per query
  ✅ Focused context = better answers
```

</details>

---

<details>

<summary><strong>📦 Why Can't I Just Send My Documents to ChatGPT directly?</strong></summary>

<br>

You can—but you'll hit walls fast.

**Context window limits 🚨** 
> GPT-4 accepts ~128k tokens. That's roughly 300 pages. Your company wiki, codebase, or document archive is likely 10x-100x larger. You physically cannot paste it all.

**Cost explosion 💥**
> Even if you could fit everything, you'd pay for every token on every query. Sending 100k tokens costs ~\$1-3 per question. Ask 50 questions a day? That's $50-150 daily—for one user.

**No selective retrieval ❌**
> When you paste documents, the model reads everything equally. It can't focus on what's relevant. Ask about refund policies and it's also processing your hiring guidelines, engineering specs, and meeting notes—wasting context and degrading answers.

**No persistence 💢**
> Every conversation starts fresh. You re-upload, re-paste, re-explain. There's no knowledge base that accumulates and improves.

</details>

---

### Why Fitz?

**Super fast setup 🐆**
> Point at a folder. Ask a question. Get an answer with sources. Everything else is handled by Fitz.

**Honest answers ✅**
> Most RAG tools confidently answer even when the answer isn't in your documents. Ask "What was our Q4 revenue?" when your docs only cover Q1-Q3, and typical RAG hallucinates a number. Fitz says: *"I cannot find Q4 revenue figures in the provided documents."*

**Swap engines, keep everything else ⚙️**
> RAG is evolving fast—GraphRAG, HyDE, ColBERT, whatever's next. Fitz lets you switch engines in one line. Your ingested data stays. Your queries stay. No migration, no re-ingestion, no new API to learn. Frameworks lock you in; Fitz lets you move.

**Analytical queries that actually work 📊**
> Standard RAG fails on questions like "What are the trends?"—it retrieves random chunks instead of insights. Fitz's hierarchical RAG generates multi-level summaries during ingestion. Ask for trends, get aggregated analysis. Ask for specifics, get detail chunks. No special syntax required.

**Other Features at a Glance 🃏**
>
>1. [x] **Local execution possible.** FAISS and Ollama support, no API keys required to start.
>2. [x] **Plugin-based architecture.** Swap LLMs, vector databases, rerankers, and retrieval pipelines via YAML config.
>3. [X] **Incremental ingestion.** Only reprocesses changed files, even with new chunking settings.
>4. [x] **Full provenance.** Every answer traces back to the exact chunk and document.
>5. [x] **Data privacy**: No telemetry, no cloud, no external calls except to the LLM provider you configure.

####

Any questions left? Try fitz on itself:

```bash
fitz quickstart ./fitz_ai "How does the chunking pipeline work?"
```

The codebase speaks for itself.

---

<details>

<summary><strong>📦 Fitz vs LangChain vs LlamaIndex</strong></summary>

<br>

#### Fitz opts for a deliberately narrower approach.
>
>LangChain and LlamaIndex are powerful **LLM application frameworks** designed to help developers build complex, end-to-end AI systems. 
>Fitz provides a **minimal, replaceable RAG engine** with strong epistemic guarantees — without locking users into a framework, ecosystem, or long-term architectural commitment.
>
>Fitz is not a competitor in scope.  
>It is an infrastructure primitive.

<br>

#### Core philosophical differences ⚖️
>
>| Dimension | Fitz | LangChain | LlamaIndex |
>|--------|------|-----------|------------|
>| Primary role | **RAG engine** | LLM application framework | LLM data framework |
>| User commitment | **No framework lock-in** | High | High |
>| Engine coupling | **Swappable in one line** | Deep | Deep |
>| Design goal | Correctness & honesty | Flexibility | Data integration |
>| Long-term risk | Low | Migration-heavy | Migration-heavy |

<br>

#### Epistemic behavior (truth over fluency) 🎯
>
>| Aspect | Fitz | LangChain / LlamaIndex |
>|-----|------|------------------------|
>| “I don’t know” | **First-class behavior** | Not guaranteed |
>| Hallucination handling | Designed-in | Usually prompt-level |
>| Confidence signaling | Explicit | Implicit |
>
>Fitz treats uncertainty as a **feature**, not a failure.  
>If the system cannot support an answer with retrieved evidence, it says so.

<br>

#### Transparency & provenance 🔎
>
>| Capability | Fitz | LangChain / LlamaIndex |
>|---------|------|------------------------|
>| Source attribution | **Mandatory** | Optional |
>| Retrieval trace | **Explicit & structured** | Often opaque |
>| Debuggability | Built-in | Tool-dependent |
>
>Every answer in Fitz is fully auditable down to the retrieval step.

<br>

#### Scope & complexity 🪐
>
>| Aspect | Fitz | LangChain / LlamaIndex |
>|-----|------|------------------------|
>| Chains / agents | ❎ | ✔ |
>| Prompt graphs | ❎ | ✔ |
>| UI abstractions | ❎ | Often |
>| Cognitive overhead | **Very low** | High |
>
>Fitz intentionally does less — so it can be trusted more.

<br>

#### Use Fitz if you want:
>
>- A replaceable RAG engine, not a framework marriage
>- Strong epistemic guarantees (“I don’t know” is valid output)
>- Full provenance for every answer
>- A transparent, extensible plugin architecture
>- A future-proof ingestion pipeline that survives engine changes

</details>

---
<details>

<summary><strong>📦 Features</strong></summary>

<br>

#### Hierarchical RAG 📊

>Standard RAG struggles with analytical queries like "What are the trends?" because it retrieves random chunks instead of aggregated insights. Hierarchical RAG solves this.
>
>**The problem ☔️**
>```
>Q: "What are the trends in my comments?"
>Standard RAG: Returns random individual comments (not useful)
>```
>
>**The solution ☀️**
>
>For documents, Fitz auto-enables hierarchy when an LLM is available. It groups by file and generates multi-level summaries:
>- **Level 0**: Original chunks (unchanged)
>- **Level 1**: Group summaries (one per source file)
>- **Level 2**: Corpus summary (aggregates all groups)
>
>**Example: YouTube comment analysis**
>```
>Ingested: 500 comments across 10 videos
>
>Level 0: "This tutorial helped me understand async/await finally!"
>Level 1: "Tutorial Video #3: 47 comments, mostly positive. Users praise
>         clarity of examples. Common request: more on error handling."
>Level 2: "Across 10 videos (500 comments): 78% positive sentiment.
>         Top themes: code clarity, pacing, example quality.
>         Recurring requests: longer videos, more advanced topics."
>```
>
>Now analytical queries retrieve summaries, while specific queries still retrieve details:
>```
>Q: "What are the overall trends in my comments?"
>→ Returns Level 2 corpus summary + Level 1 video summaries
>```
>```
>Q: "What did people say about my async tutorial?"
>→ Returns Level 0 individual comments from that video
>```
>
>No special query syntax. No retrieval config changes. Summaries match analytical queries naturally via vector similarity.

<br>

#### Actually admits when it doesn't know 📚

> When documents don't contain the answer, fitz says so:
>
> ```
> Q: "What was our Q4 revenue?"
> A: "I cannot find Q4 revenue figures in the provided documents.
>     The available financial data covers Q1-Q3 only."
>
>    Mode: ABSTAIN
>```
>
>Three constraint plugins run automatically:
>1. [X] **📕 ConflictAwareConstraint**: Detects contradictions across sources
>2. [X] **📗 InsufficientEvidenceConstraint**: Blocks answers without evidence
>3. [X] **📘 CausalAttributionConstraint**: Prevents hallucinated cause-effect claims

<br>

#### Swappable RAG Engines 🔄

>Your data stays. Your queries stay. Only the engine changes.
>
>```
>        ┌─────────────────────────────────────┐
>        │           Your Query                │
>        │   "What are the payment terms?"     │
>        └──────────────────┬──────────────────┘
>                           │
>                           ▼
>        ┌─────────────────────────────────────┐
>        │       engine="..."                  │
>        │  ┌─────────┐ ┌───────┐ ┌─────────┐  │
>        │  │ classic │ │ clara │ │ graph   │  │
>        │  │  _rag   │ │       │ │  _rag   │  │
>        │  └────┬────┘ └───┬───┘ └────┬────┘  │
>        │       └──────────┼──────────┘       │
>        └──────────────────┼──────────────────┘
>                           │
>                           ▼
>        ┌─────────────────────────────────────┐
>        │       Your Ingested Knowledge       │
>        │      (unchanged across engines)     │
>        └─────────────────────────────────────┘
>```
>
>```python
>answer = run("What are the payment terms?", engine="classic_rag")
>answer = run("What are the payment terms?", engine="clara")
>answer = run("What are the payment terms?", engine="graph_rag")  # future
>```
>
>No migration. No re-ingestion. No new API to learn.

<br>

#### Full Provenance 🗂️

>Every answer traces back to its source:
>
>```
>Answer: The refund policy allows returns within 30 days...
>
>Sources:
>  [1] policies/refund.md [chunk 3] (score: 0.92)
>  [2] faq/payments.md [chunk 1] (score: 0.87)
>```

<br>

#### Incremental Ingestion ⚡

>Fitz tracks file hashes and only re-ingests what changed:
>
>```
>$ fitz ingest ./src
>
>Scanning... 847 files
>  → 12 new files
>  → 3 modified files
>  → 832 unchanged (skipped)
>
>Ingesting 15 files...
>```
>
>Re-running ingestion on a large codebase takes seconds, not minutes. Changed your chunking config? Fitz detects that too and re-processes affected files.

<br>

#### Smart Chunking 🧠

>Format-aware chunking that preserves structure:
>
>| Format | Strategy |
>|--------|----------|
>| **Python** | AST-aware: keeps classes, functions, imports intact. Large classes split by method. |
>| **Markdown** | Header-aware: splits on `#` headers, preserves code blocks and lists. Extracts YAML frontmatter as metadata. |
>| **PDF** | Section-aware: detects numbered headings (1.1, 2.3.1), roman numerals, and keywords (Abstract, Conclusion). |
>
>No more retrieving half a function or a code block split mid-syntax.

</details>

---

<details>

<summary><strong>📦 Quick Start</strong></summary>

<br>

```bash
pip install fitz-ai

fitz quickstart ./docs "Your question here"
```

That's it. Fitz will prompt you for anything it needs.

<br>

Want to go fully local with Ollama? No problem:

```bash
pip install fitz-ai[local]

ollama pull llama3.2
ollama pull nomic-embed-text

fitz quickstart ./docs "Your question here"
```

No data leaves your machine. No API costs. Same interface.

</details>

---

<details>

<summary><strong>📦 Real-World Usage</strong></summary>

<br>

Fitz is a foundation. It handles document ingestion and grounded retrieval—you build whatever sits on top: chatbots, dashboards, alerts, or automation.

<br>

<strong>Chatbot Backend 🤖</strong>

> Connect fitz to Slack, Discord, Teams, or your own UI. One function call returns an answer with sources—no hallucinations, full provenance. You handle the conversation flow; fitz handles the knowledge.
>
> *Example:* A SaaS company plugs fitz into their support bot. Tier-1 questions like "How do I reset my password?" get instant answers. Their support team focuses on edge cases while fitz deflects 60% of incoming tickets.

<br>

<strong>Internal Knowledge Base 📖</strong>

> Point fitz at your company's wiki, policies, and runbooks. Employees ask natural language questions instead of hunting through folders or pinging colleagues on Slack.
>
> *Example:* A 200-person startup ingests their Notion workspace and compliance docs. New hires find answers to "How do I request PTO?" on day one—no more waiting for someone in HR to respond.

<br>

<strong>Continuous Intelligence & Alerting (Watchdog) 🐶</strong>

> Pair fitz with cron, Airflow, or Lambda. Ingest data on a schedule, run queries automatically, trigger alerts when conditions match. Fitz provides the retrieval primitive; you wire the automation.
>
> *Example:* A security team ingests SIEM logs nightly. Every morning, a scheduled job asks "Were there failed logins from unusual locations?" If fitz finds evidence, an alert fires to the on-call channel before anyone checks email.

<br>

<strong>Web Knowledge Base 🌎</strong>

> Scrape the web with Scrapy, BeautifulSoup, or Playwright. Save to disk, ingest with fitz. The web becomes a queryable knowledge base.
>
> *Example:* A football analytics hobbyist scrapes Premier League match reports. After ingesting, they ask "How did Arsenal perform against top 6 teams?" or "What tactics did Liverpool use in away games?"—insights that would take hours to compile manually.

<br>

<strong>Codebase Search 🐍</strong>

> Fitz includes built-in AST-aware chunking for code bases. Functions, classes, and modules become individual searchable units with docstrings and imports preserved. Ask questions in natural language; get answers pointing to specific code.
>
> *Example:* A team inherits a legacy Django monolith—200k lines, sparse docs. They ingest the codebase and ask "Where is user authentication handled?" or "What API endpoints modify the billing table?" New developers onboard in days instead of weeks.

</details>

---

<details>

<summary><strong>📦 Architecture</strong></summary>

<br>

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
│  code artifacts | LLM summaries | hierarchical RAG | custom   │
├───────────────────────────────────────────────────────────────┤
│  Constraints (epistemic safety)                               │
│  ConflictAware | InsufficientEvidence | CausalAttribution     │
└───────────────────────────────────────────────────────────────┘
```

</details>

---

<details>

<summary><strong>📦 CLI Reference</strong></summary>

<br>

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

</details>

---

<details>

<summary><strong>📦 Beyond RAG</strong></summary>

<br>

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

</details>

---

<details>

<summary><strong>📦 Philosophy</strong></summary>

<br>

**Principles:**
- **Explicit over clever**: No magic. Read the config, know what happens.
- **Answers over architecture**: Optimize for time-to-insight, not flexibility.
- **Honest over helpful**: Better to say "I don't know" than hallucinate.
- **Files over frameworks**: YAML plugins over class hierarchies.

</details>

---

### License

MIT

---

### Links

- [GitHub](https://github.com/yafitzdev/fitz-ai)
- [PyPI](https://pypi.org/project/fitz-ai/)
- [Changelog](CHANGELOG.md)
- [CLI Documentation](docs/CLI.md)
