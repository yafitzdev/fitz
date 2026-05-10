

<div align="center">

# fitz-sage

### The RAG library that says "I don't know" instead of hallucinating.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-sage.svg)](https://pypi.org/project/fitz-sage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.11.0-green.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/yafitzdev/fitz-sage)

[Why `fitz-sage`?](#why-fitz) • [Retrieval Intelligence](#retrieval-intelligence) • [Governance](#governance--know-what-you-dont-know) • [Documentation](#links) • [GitHub](https://github.com/yafitzdev/fitz-sage)

</div>

<br />

---

<div align="center">
<table>
  <tr>
    <td align="center" colspan="2">
      <pre><strong>Q: "Who won the 2024 FIFA World Cup?"</strong>
(There was no World Cup in 2024.)</pre>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>❌ Uncalibrated RAG systems</strong>
<pre>
A: "Germany won the 2024 FIFA World Cup,
    defeating Argentina 1-0 in the final."
</pre>
    </td>
    <td align="center" width="50%">
      <strong>🛡️ fitz-sage</strong>
<pre>
A: "I don't have enough information
    to answer this question.
</pre><pre>
    Related topics in the knowledge base:
      - FIFA tournament history (4 mentions)
      - 2022 World Cup coverage (7 mentions)
</pre><pre>
    To answer this, consider adding:
      - Documents covering 2024 FIFA events."
</pre>
    </td>
  </tr>
</table>
  → Uncalibrated RAG hallucinates confidently when the answer isn't in your documents. 
  
  `fitz-sage` refuses, explains why, and tells you what to add.
</div>



---

### Where to start 🚀

> [!IMPORTANT]
> Requires **any OpenAI-compatible LLM endpoint** — local ([llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`, [vLLM](https://github.com/vllm-project/vllm), LM Studio, Ollama in `/v1/` mode) or cloud (OpenAI, Together, Groq, Fireworks, OpenRouter, …). `fitz-sage` auto-detects a local server on the standard ports (8080 / 8000 / 1234 / 11434) on first run, or falls back to `OPENAI_API_KEY`.

```bash
pip install fitz-sage

# Local (recommended): start llama-server with any GGUF chat model
llama-server -m qwen2.5-7b-instruct.gguf --port 8080 &

# Then point fitz-sage at it — same syntax for cloud:
fitz query "What is our refund policy?" --source ./docs
```

That's it. Your documents are now searchable with AI.

> [!TIP]
> **Single config knob.** No more "find a provider that has both chat *and* embeddings", no more swapping models mid-query. One URL, one model spec, done — local or cloud:
>
> ```bash
> # Override config at the CLI for any OpenAI-compatible URL:
> fitz query "..." --endpoint http://localhost:8080/v1 --model qwen2.5-7b
> fitz query "..." --endpoint https://api.together.xyz/v1 \
>                  --model meta-llama-3.1-70b \
>                  --api-key-env TOGETHER_API_KEY
> ```


![fitz-sage quickstart demo](https://raw.githubusercontent.com/yafitzdev/fitz-sage/main/docs/assets/quickstart_demo.gif)
*Figure 1: Example of user experience for querying documents using fitz-sage.*

---

### About

Existing RAG tools hallucinate. When the answer isn't in your documents, they invent one — confidently, fluently, wrongly. 
In production, that's not a minor inconvenience. It's the reason you can't trust the system. I built fitz-sage to solve that problem directly, while working as a Data Engineer in the automotive industry. No LangChain. No LlamaIndex. Every layer written from scratch.

The retrieval architecture is [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md) — documents are parsed into typed units (
code symbols, sections, tables) and each query is routed to the right search strategy, rather than searching flat chunks uniformly.

Honesty is enforced by an [ML governance classifier](docs/features/governance/governance-benchmarking.md) that decides when to answer, hedge, refuse — validated against 
[fitz-gov](https://github.com/yafitzdev/fitz-gov), a purpose-built benchmark of 2,920 adversarial test cases (5.7% false-trustworthy rate).

It runs in production today and powers [fitz-forge](https://github.com/yafitzdev/fitz-forge).

A single chat-protocol implementation (OpenAI HTTP). 2,000+ tests. 99% coverage.

Yan Fitzner — ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev)).

![fitz-sage honest_rag](https://raw.githubusercontent.com/yafitzdev/fitz-sage/main/docs/assets/honest_rag.jpg)

---

<details>

<summary><strong>📦 What is RAG?</strong></summary>

<br>

RAG is how ChatGPT's "file search," Notion AI, and enterprise knowledge tools actually work under the hood.
Instead of sending all your documents to an AI, RAG:

1. [X] **Indexes your documents** — Splits them into chunks, converts to vectors, stores in a database
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
> GPT-4 accepts ~128k tokens. That's roughly 300 pages. Your company wiki, codebase, or document archive is likely 10x-100x 
> larger. You physically cannot paste it all.

**Cost explosion 💥**
> Even if you could fit everything, you'd pay for every token on every query. Sending 100k tokens costs ~\$1-3 per question. 
> Ask 50 questions a day? That's $50-150 daily—for one user.

**No selective retrieval ❌**
> When you paste documents, the model reads everything equally. It can't focus on what's relevant. Ask about refund policies 
> and it's also processing your hiring guidelines, engineering specs, and meeting notes—wasting context and degrading answers.

**No persistence 💢**
> Every conversation starts fresh. You re-upload, re-paste, re-explain. There's no knowledge base that accumulates and improves.

</details>

---

<details>

<summary><strong>📦 How is this different from LangChain / LlamaIndex?</strong></summary>

<br>

They're frameworks — you assemble the chunker, embedder, vector store, retriever, and prompt chain yourself. fitz-sage is 
a library — one function call that handles all of it with built-in intelligence.

You trade flexibility for a pipeline that handles temporal queries, comparison queries, code symbol extraction, tabular 
SQL, and epistemic honesty out of the box — without configuration.

</details>

---

### Why `fitz-sage`?

**Asymmetric indexing 🗂️** → [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md)
> Documents are parsed into typed retrieval units (symbols, sections, tables) with structural metadata, not flat chunks. 
> Queries are routed to the right strategy per content type.

**Zero-wait querying 🐆** → [Progressive KRAG](docs/features/platform/progressive-krag-agentic-search.md)
> Ask a question immediately — no ingestion step required. `fitz-sage` serves answers instantly via agentic search while
> a background worker indexes your files. Queries get faster over time as indexing completes, but they work from second one.

**Honest answers ✅** → [Governance Benchmark](docs/features/governance/governance-benchmarking.md)
> Most RAG tools confidently answer even when the answer isn't in your documents. Ask "What was our Q4 revenue?" when 
> your docs only cover Q1-Q3, and typical RAG hallucinates a number. `fitz-sage` says: *"I cannot find Q4 revenue figures 
> in the provided documents."
>
> → Detects when to abstain at **86.5% recall** on [fitz-gov](https://github.com/yafitzdev/fitz-gov), a 2,920 case benchmark for epistemic honesty (62.7% hard difficulty). 
> False-trustworthy rate: **5.7%**.

**Actionable failures 🔍**
> When `fitz-sage` can't answer, it doesn't just refuse — it explains what it searched for, shows related topics that *do* 
> exist, and suggests what documents to add. When sources conflict, it tells you exactly which sources disagree and 
> what the disagreement is about. Every failure mode is a feedback signal, not a dead end.

**Queries that actually work 📊**
> Standard RAG fails silently on real queries. `fitz-sage` has built-in intelligence: hierarchical summaries for "What are the trends?", 
> exact keyword matching for "Find TC-1000", multi-query decomposition for complex questions, address-based code retrieval with 
> import graph traversal, and SQL execution for tabular data. No configuration—it just works.

**Tabular data that is actually searchable 📈** → [Unified Storage](docs/features/platform/unified-storage.md)
> CSV and table data is a nightmare in most RAG systems—chunked arbitrarily, structure lost, queries fail. `fitz-sage` 
> stores tables natively in PostgreSQL alongside your vectors—same database, no sync issues. Auto-detects schema and runs 
> real SQL. Ask "What's the average price by region?" and get an actual computed answer, not fragmented rows.

**Fully local execution possible 🏠** → [OpenAI-Compatible Endpoint](docs/features/platform/openai-compatible-endpoint.md)
> Embedded PostgreSQL + any local OpenAI-compatible server (llama.cpp, vLLM, LM Studio, Ollama). One protocol, one URL, no API keys required to start.

####

> [!TIP]
> Any questions left? Try fitz on itself:
>
> ```bash
> fitz query "How does the retrieval pipeline work?" --source ./fitz_sage
> ```
>
> The codebase speaks for itself.

---

### What You Can Search

Traditional RAG chops every document into flat text blocks and searches them the same way. [FitzKRAG](docs/features/platform/krag.md) parses each 
document by type — tree-sitter for code, heading hierarchy for docs, schema detection for CSVs — and produces typed retrieval 
units, each with its own storage format and search strategy.

<br>

| Retrieval Unit              | Extracted From | How It Works |
|-----------------------------|----------------|-------------|
| [**Symbols 🖌️**](docs/features/ingestion/code-symbol-extraction.md) | Code files | Tree-sitter parses functions, classes, and methods into addressable units with qualified names, references, and import graphs. Cross-file dependencies are graph traversals, not text searches. |
| **Sections 📑** | Documents (PDF, markdown, text) | Headings and paragraphs are extracted with parent/child hierarchy. Deeply nested sections include parent context; top-level headings include child summaries. |
| [**Tables 📅**](docs/features/ingestion/tabular-data-routing.md) | CSV files or tables within documents | Native PostgreSQL storage with auto-detected schema. Real SQL execution from natural language — not chunked text. |
| **Images 🖼️** | Figures and diagrams within documents | VLM-powered figure extraction and visual understanding. *(Coming soon)* |
| **Chunks 🧩** | Any content as fallback | Traditional chunk-based retrieval when structured extraction doesn't apply. Automatic fallback — no configuration needed. |

<br>

> [!NOTE]
> All retrieval units share the same retrieval intelligence (temporal handling, comparison queries, multi-hop reasoning, etc.) 
> and the same enrichment pipeline (summaries, keywords, entities, hierarchical summaries).

---

### Retrieval Intelligence

Most RAG implementations are naive vector search—they fail silently on real-world queries. 
`fitz-sage` has [built-in intelligence](docs/features/retrieval) that handles edge cases automatically:

<br>

| Feature | Query | Naive RAG Problem | `fitz-sage` Solution |
|---------|-------|-------------------|------------------|
| [**epistemic-honesty**](docs/features/governance/epistemic-honesty.md) | "What was our Q4 revenue?" | ❌ Hallucinated number — Info doesn't exist, but LLM won't admit it | ✅ "I don't know" |
| [**keyword-vocabulary**](docs/features/retrieval/keyword-vocabulary.md) | "Find TC_1000" | ❌ Wrong test case — Embeddings see TC_1000 ≈ TC_2000 (semantically similar) | ✅ Exact keyword matching |
| [**hybrid-search**](docs/features/retrieval/hybrid-search.md) | "X100 battery specs" | ❌ Returns Y200 docs — Semantic search misses exact model numbers | ✅ Hybrid search (dense + sparse) |
| [**sparse-search**](docs/features/retrieval/sparse-search.md) | "error code E_AUTH_401" | ❌ No exact match — Embeddings miss precise error codes | ✅ PostgreSQL full-text search |
| [**multi-hop**](docs/features/retrieval/multi-hop-reasoning.md) | "Who wrote the paper cited by the 2023 review?" | ❌ Returns the review only — Single-step search can't traverse references | ✅ Iterative retrieval |
| [**hierarchical-rag**](docs/features/ingestion/hierarchical-rag.md) | "What are the design principles?" | ❌ Random fragments — Answer is spread across docs; no single chunk contains it | ✅ Hierarchical summaries |
| [**multi-query**](docs/features/retrieval/multi-query-rag.md) | *[User pastes 500-char test report]* "What failed and why?" | ❌ Vaguely related chunks — Long input → averaged embedding → matches nothing specifically | ✅ Multi-query decomposition |
| [**comparison-queries**](docs/features/retrieval/comparison-queries.md) | "Compare React vs Vue performance" | ❌ Incomplete comparison — Only retrieves one entity, missing the other | ✅ Multi-entity retrieval |
| [**entity-graph**](docs/features/retrieval/entity-graph.md) | "What else mentions AuthService?" | ❌ Isolated chunks — No awareness of shared entities across docs | ✅ Entity-based linking across sources |
| [**temporal-queries**](docs/features/retrieval/temporal-queries.md) | "What changed between Q1 and Q2?" | ❌ Random chunks — No awareness of time periods in query | ✅ Temporal query handling |
| [**aggregation-queries**](docs/features/retrieval/aggregation-queries.md) | "List all the test cases that failed" | ❌ Partial list — No mechanism for comprehensive retrieval | ✅ Aggregation query handling |
| [**freshness-authority**](docs/features/retrieval/freshness-authority.md) | "What does the official spec say?" | ❌ Returns notes — Can't distinguish authoritative vs informal sources | ✅ Freshness/authority boosting |
| [**query-expansion**](docs/features/retrieval/query-expansion.md) | "How do I fetch the db config?" | ❌ No matches — User says "fetch", docs say "retrieve"; "db" vs "database" | ✅ Query expansion |
| [**query-rewriting**](docs/features/retrieval/query-rewriting.md) | "Tell me more about it" *(after discussing TechCorp)* | ❌ Lost context — Pronouns like "it" reference nothing, retrieval fails | ✅ Conversational context resolution |
| [**hyde**](docs/features/retrieval/hyde.md) | "What's TechCorp's approach to sustainability?" | ❌ Poor recall — Abstract queries don't embed close to concrete documents | ✅ Hypothetical document generation |
| [**contextual-embeddings**](docs/features/retrieval/contextual-embeddings.md) | "When does it expire?" | ❌ Ambiguous chunk — "It expires in 24h" embedded without context; "it" = ? | ✅ Summary-prefixed symbol/section embeddings |
| [**reranking**](docs/features/retrieval/reranking.md) | "What's the battery warranty?" | ❌ Imprecise ranking — Vector similarity ≠ true relevance; best answer buried | ✅ Cross-encoder precision |

<br>

> [!IMPORTANT]
> These features are **always on**—no configuration needed. `fitz-sage` automatically detects when to use each capability.

---

### Governance — Know What You Don't Know

[Feature docs](docs/features/governance/governance-benchmarking.md) • [fitz-gov benchmark](https://github.com/yafitzdev/fitz-gov)

Most RAG systems hallucinate confidently. `fitz-sage` **measures and enforces** epistemic honesty using a 5-question cascade ML 
classifier trained on 2,920 labeled cases from [fitz-gov](https://github.com/yafitzdev/fitz-gov), a benchmark for epistemic honesty.

<br>

```
  Query + Retrieved Context
            │
            ▼
  ┌─────────────────────┐
  │ 5 Constraints       │     Contradiction detection, evidence sufficiency,
  │ (epistemic sensors) │     causal attribution, answer verification, specific info type
  └──────────┬──────────┘
             │ 108 features extracted
             ▼
  ┌─────────────────────┐
  │ Q1: Evidence        │     Is the evidence sufficient?
  │ sufficient? (ML)    ├───► NO ──► ABSTAIN
  └──────────┬──────────┘
             │ YES
             ▼
  ┌─────────────────────┐
  │ Q2: Conflict?       │     Is there material conflict? (ML router)
  │ (ML)                ├───► YES ──┐
  └──────────┬──────────┘           │
             │ NO                   ▼
             │            ┌─────────────────────┐
             │            │ Q3: Conflict        │
             │            │ resolved? (ML)      ├───► NO ──► DISPUTED
             │            └──────────┬──────────┘
             │                       └ YES ────────────────► TRUSTWORTHY
             ▼
  ┌─────────────────────┐
  │ Q4: Evidence truly  │     Is the evidence solid enough?
  │ solid? (ML)         ├───► NO ──► ABSTAIN
  └──────────┬──────────┘
             └ YES ────────────────► TRUSTWORTHY

```

<br>

| Decision | Meaning                              | Recall    |
|----------|--------------------------------------|-----------|
| **ABSTAIN** | Evidence doesn't answer the question | **86.5%** |
| **DISPUTED** | Sources contradict each other        | **86.1%** |
| **TRUSTWORTHY** | Consistent, sufficient evidence      | **70.0%** |

**Overall accuracy: 78.7%** | **False-trustworthy: 5.7%** on fitz-gov (2,920 cases, 5-fold cross-validated, 62.7% hard difficulty)

<br>

> [!NOTE]
> Governance asks "given three relevant documents that partially contradict each other, should you flag a dispute, hedge 
> the answer, or trust the consensus?" That's a judgment call even humans disagree on.

<strong>The system fails safe 🛡️</strong>
> The safety-first threshold is tuned so that when the classifier is wrong, it over-hedges ("disputed" instead of "trustworthy") — 
> annoying but harmless. Over-confidence ("trustworthy" instead of "disputed") is the rarest error mode.

<strong>These scores are a floor, not a ceiling 👣</strong>
> All benchmarks were measured using `qwen2.5:3b` — a 3B parameter local model. The governance constraints run on the 
> fast-tier LLM to keep latency low. Stronger models produce better constraint signals, which feed better features into the classifier. 
> Upgrading your chat provider should improve governance accuracy for free.

<strong>Zero extra latency ⏱️</strong>
> The constraints already run as part of the pipeline. The ML classifier just replaces hand-coded rules with a local sklearn model — 
> inference takes microseconds, no additional API calls.

---

<details>

<summary><strong>📦 Quick Start</strong></summary>

<br>

#### CLI
>
>```bash
>pip install fitz-sage
>
>fitz query "Your question here" --source ./docs
>```
>
>`fitz-sage` auto-detects your LLM provider on first run:
>1. **Local OpenAI-compatible server running?** → Uses it automatically (probes ports 8080 / 8000 / 1234 / 11434 for `/v1/models`)
>2. **`OPENAI_API_KEY` set?** → Uses it automatically
>3. **Neither?** → Prints actionable setup instructions (start `llama-server`, set `OPENAI_API_KEY`, or use `--endpoint`)
>
>For one-off queries against any OpenAI-compatible URL, skip the config:
>
>```bash
>fitz query "..." --endpoint http://localhost:8080/v1 --model qwen2.5-7b
>fitz query "..." --endpoint https://api.together.xyz/v1 \
>                 --model meta-llama-3.1-70b \
>                 --api-key-env TOGETHER_API_KEY
>```

<br>

#### Python SDK
>
>```python
>import fitz_sage
>
>answer = fitz_sage.query("Your question here", source="./docs")
>
>print(answer.text)
>for source in answer.provenance:
>    print(f"  - {source.source_id}: {source.excerpt[:50]}...")
>```
>
>The SDK provides:
>- Module-level `query()` matching CLI
>- Auto-config creation (no setup required)
>- Full provenance tracking
>- Same honest retrieval as the CLI
>
>For advanced use (multiple collections), use the `fitz` class directly:
>```python
>from fitz_sage import fitz
>
>physics = fitz(collection="physics")
>answer = physics.query("Explain entanglement", source="./physics_papers")
>```

<br>

#### Fully Local (llama.cpp recommended)
>
>```bash
>pip install fitz-sage
>
># Chat server on port 8080 — fitz-sage's default chat endpoint
>llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080 -c 8192
>
># Embedding server on port 8081 — fitz-sage's default embedding endpoint
>llama-server -m nomic-embed-text-v1.5.Q4_K_M.gguf --port 8081 --embeddings
>
>fitz query "Your question here" --source ./docs
>```
>
>Both models stay hot in their own processes — no model thrashing, no SDK juggling, no second API key. Auto-detection picks them up from the standard ports. No data leaves your machine.
>
>Other compatible servers: [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.ai) (in `/v1/` mode), [TabbyAPI](https://github.com/theroyallab/tabbyAPI). Anything that speaks the OpenAI HTTP protocol works.

</details>

---

<details>

<summary><strong>📦 Real-World Usage</strong></summary>

<br>

`fitz-sage` is a foundation. It handles document indexing and grounded retrieval—you build whatever sits on top: chatbots, dashboards, alerts, or automation.

<br>

<strong>Chatbot Backend 🤖</strong>

> Connect fitz to Slack, Discord, Teams, or your own UI. One function call returns an answer with sources—no 
> hallucinations, full provenance. You handle the conversation flow; fitz handles the knowledge.
>
> *Example:* A SaaS company plugs fitz into their support bot. Tier-1 questions like "How do I reset my password?" get 
> instant answers. Their support team focuses on edge cases while fitz deflects 60% of incoming tickets.

<br>

<strong>Internal Knowledge Base 📖</strong>

> Point fitz at your company's wiki, policies, and runbooks. Employees ask natural language questions instead of hunting 
> through folders or pinging colleagues on Slack.
>
> *Example:* A 200-person startup points fitz at their Notion workspace and compliance docs. New hires find answers to 
> "How do I request PTO?" on day one—no more waiting for someone in HR to respond.

<br>

<strong>Continuous Intelligence & Alerting (Watchdog) 🐶</strong>

> Pair fitz with cron, Airflow, or Lambda. Point at data on a schedule, run queries automatically, trigger alerts when 
> conditions match. `fitz-sage` provides the retrieval primitive; you wire the automation.
>
> *Example:* A security team points fitz at SIEM logs nightly. Every morning, a scheduled job asks "Were there failed 
> logins from unusual locations?" If fitz finds evidence, an alert fires to the on-call channel before anyone checks email.

<br>

<strong>Web Knowledge Base 🌎</strong>

> Scrape the web with Scrapy, BeautifulSoup, or Playwright. Save to disk, point fitz at it. The web becomes a queryable 
> knowledge base.
>
> *Example:* A football analytics hobbyist scrapes Premier League match reports. They point fitz at the folder and ask 
> "How did Arsenal perform against top 6 teams?" or "What tactics did Liverpool use in away games?"—insights that would 
> take hours to compile manually.

<br>

<strong>Codebase Search 🐍</strong> → [Code Symbol Extraction](docs/features/ingestion/code-symbol-extraction.md) • [KRAG](docs/features/platform/krag.md)

> **Two modes of code retrieval:**
>
> **Full KRAG** — tree-sitter parses your codebase into symbols (functions, classes, methods) with qualified names, 
> references, and import graphs. No chunking—each symbol is a precise, addressable unit. Cross-file dependencies are 
> tracked, so "what calls this function?" is a graph traversal, not a text search.
>
> **Standalone (`pip install fitz-sage[code]`)** — Zero-dependency code retrieval via `CodeRetriever`. Builds an AST 
> structural index, uses an LLM to select relevant files, expands via import graph and neighbor directories, and returns 
> compressed results. No PostgreSQL, no pgvector, no docling—just point at a directory and ask.
>
> *Example:* A team inherits a legacy Django monolith—200k lines, sparse docs. They point fitz at the codebase and ask 
> "Where is user authentication handled?" or "What depends on the billing module?" FitzKRAG returns specific functions with 
> their callers and dependencies. New developers onboard in days instead of weeks.

</details>

---

<details>

<summary><strong>📦 Architecture</strong> → <a href="docs/ARCHITECTURE.md">Full Architecture Guide</a></summary>

<br>

```
┌───────────────────────────────────────────────────────────────┐
│                         fitz-sage                               │
├───────────────────────────────────────────────────────────────┤
│  User Interfaces                                              │
│  CLI: query (--source) | init | collections | config | serve  │
│  SDK: fitz_sage.query(source=...)                                │
│  API: /query | /chat | /collections | /health                 │
├───────────────────────────────────────────────────────────────┤
│  Engines                                                      │
│  ┌────────────┐  ┌────────────┐                               │
│  │  FitzKRAG  │  │  Custom... │  (extensible registry)        │
│  └────────────┘  └────────────┘                               │
├───────────────────────────────────────────────────────────────┤
│  LLM Providers (single OpenAI-compatible HTTP protocol)       │
│  ┌────────┐ ┌───────────┐ ┌──────────────┐                    │
│  │  Chat  │ │ Embedding │ │ LLM Reranker │                    │
│  └────────┘ └───────────┘ └──────────────┘                    │
│  endpoint/<URL> | openai | azure_openai | enterprise          │
├───────────────────────────────────────────────────────────────┤
│  Storage (PostgreSQL + pgvector)                              │
│  vectors | metadata | tables | keywords | full-text search    │
├───────────────────────────────────────────────────────────────┤
│  Retrieval (address-based, baked-in intelligence)             │
│  symbols | sections | tables | import graphs | reranking      │
├───────────────────────────────────────────────────────────────┤
│  Enrichment (baked in)                                        │
│  summaries | keywords | entities | hierarchical summaries     │
├───────────────────────────────────────────────────────────────┤
│  Constraints (epistemic safety)                               │
│  ConflictAware | InsufficientEvidence | CausalAttribution     │
└───────────────────────────────────────────────────────────────┘
```

</details>

---

<details>

<summary><strong>📦 CLI Reference</strong> → <a href="docs/CLI.md">Full CLI Guide</a></summary>

<br>

```bash
fitz query "question" --source ./docs  # Point at docs and query (start here)
fitz query "question"                  # Query existing collection
fitz query --chat                      # Multi-turn conversation mode
fitz collections                       # List and delete knowledge collections
fitz serve                             # Start REST API server
fitz reset                             # Reset pgserver database (when stuck/corrupted)
```

Config: `.fitz/config.yaml` — auto-created on first run, edit to change models.

</details>

---

<details>

<summary><strong>📦 Python SDK Reference</strong> → <a href="docs/SDK.md">Full SDK Guide</a></summary>

<br>

**Simple usage (module-level, matches CLI):**
```python
import fitz_sage

answer = fitz_sage.query("What is the refund policy?", source="./docs")
print(answer.text)
```

<br>

**Advanced usage (multiple collections):**
```python
from fitz_sage import fitz

# Create separate instances for different collections
physics = fitz(collection="physics")
legal = fitz(collection="legal")

# Query each collection
physics_answer = physics.query("Explain entanglement", source="./physics_papers")
legal_answer = legal.query("What are the payment terms?", source="./contracts")
```

<br>

**Working with answers:**
```python
answer = fitz_sage.query("What is the refund policy?")

print(answer.text)
print(answer.mode)  # TRUSTWORTHY, DISPUTED, or ABSTAIN

for source in answer.provenance:
    print(f"Source: {source.source_id}")
    print(f"Excerpt: {source.excerpt}")
```

</details>

---

<details>

<summary><strong>📦 REST API Reference</strong> → <a href="docs/API.md">Full API Guide</a></summary>

<br>

**Start the server:**
```bash
pip install fitz-sage[api]

fitz serve                    # localhost:8000
fitz serve -p 3000            # custom port
fitz serve --host 0.0.0.0     # all interfaces
```

**Interactive docs:** Visit `http://localhost:8000/docs` for Swagger UI.

<br>

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Query knowledge base |
| POST | `/chat` | Multi-turn chat (stateless) |
| GET | `/collections` | List all collections |
| GET | `/collections/{name}` | Get collection stats |
| DELETE | `/collections/{name}` | Delete a collection |
| GET | `/health` | Health check |

<br>

**Example request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "collection": "default"}'
```

</details>

---

<details>

<summary><strong>📦 FAQ / Troubleshooting</strong></summary>

<br>

**`fitz` command not found after install**
> Your Python Scripts directory isn't on PATH. Use `python -m fitz_sage.cli.cli` instead, or add the Scripts directory 
> to your system PATH.

**PDF/DOCX files are being skipped**
> Document parsing requires docling, which is optional to keep the base install lightweight. Install it with: 
> `pip install fitz-sage[docs]`

**"Connection refused at localhost:8080" error**
> No OpenAI-compatible server is running. Start one — for example with llama.cpp:
> `llama-server -m model.gguf --port 8080 -c 8192`. Or override the URL at the CLI:
> `fitz query "..." --endpoint https://api.openai.com/v1 --api-key-env OPENAI_API_KEY`.

**"Model not found" error**
> The model name in your config doesn't match what your server has loaded. Check `/v1/models` on your server:
> `curl http://localhost:8080/v1/models`. Then update `chat_smart` in `.fitz/config.yaml` to match.

**First query is slow**
> First run initializes the database and warms up the LLM. Subsequent queries are much faster. Local models load
> on first use — use a smaller GGUF (e.g. Qwen2.5-1.5B) for faster cold start, or run llama-server in advance.

**How do I change my LLM endpoint or model?**
> Edit `.fitz/config.yaml`:
> ```yaml
> chat_smart: endpoint/qwen2.5-7b-instruct
> chat_base_url: http://localhost:8080/v1
> embedding: endpoint/nomic-embed-text-v1.5
> embedding_base_url: http://localhost:8081/v1
> ```
> Or override at the CLI without editing YAML:
> ```bash
> fitz query "..." --endpoint http://localhost:8080/v1 --model qwen2.5-7b
> ```

**How do I use a cloud provider?**
> Either use the `openai` preset (built-in OpenAI URL):
> ```yaml
> chat_smart: openai/gpt-4o
> embedding: openai/text-embedding-3-small
> # OPENAI_API_KEY in env
> ```
> Or any OpenAI-compatible cloud via the `endpoint` provider:
> ```yaml
> chat_smart: endpoint/meta-llama-3.1-70b
> chat_base_url: https://api.together.xyz/v1
> chat_api_key_env: TOGETHER_API_KEY
> ```
> See [docs/features/platform/openai-compatible-endpoint.md](docs/features/platform/openai-compatible-endpoint.md) for a migration table from the older Ollama / Cohere / Anthropic provider names.

**How do I reset everything?**
> Delete the `.fitz/` directory in your project root. Next run will re-detect and re-configure.

</details>

---

### License

MIT

---

### Links

- [GitHub](https://github.com/yafitzdev/fitz-sage)
- [PyPI](https://pypi.org/project/fitz-sage/)
- [Changelog](CHANGELOG.md)

**Documentation:**
- [CLI Reference](docs/CLI.md)
- [Python SDK](docs/SDK.md)
- [REST API](docs/API.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Unified Storage (PostgreSQL + pgvector)](docs/features/platform/unified-storage.md)
- [Progressive KRAG & Agentic Search](docs/features/platform/progressive-krag-agentic-search.md)
- [Ingestion Pipeline](docs/INGESTION.md)
- [Enrichment (Hierarchies, Entities)](docs/ENRICHMENT.md)
- [Epistemic Constraints](docs/CONSTRAINTS.md)
- [Governance Benchmarking (fitz-gov)](docs/features/governance/governance-benchmarking.md)
- [BEIR Benchmark Results](docs/evaluation/beir-results.md)
- [Plugin Development](docs/PLUGINS.md)
- [Feature Control](docs/FEATURE_CONTROL.md)
- [KRAG — Knowledge Routing Augmented Generation](docs/features/platform/krag.md)
- [Code Symbol Extraction](docs/features/ingestion/code-symbol-extraction.md)
- [Tabular Data Routing](docs/features/ingestion/tabular-data-routing.md)
- [Standalone Code Retrieval](docs/features/code/standalone-code-retrieval.md)
- [Enterprise Gateway](docs/features/platform/enterprise-gateway.md)
- [Engines](docs/ENGINES.md)
- [Configuration Examples](docs/CONFIG_EXAMPLES.md)
- [Custom Engines](docs/CUSTOM_ENGINES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
