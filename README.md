<!-- README.md -->

<div align="center">

# fitz-sage

### The RAG library that says "I don't know" instead of hallucinating.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-sage.svg)](https://pypi.org/project/fitz-sage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.14.1-green.svg)](CHANGELOG.md)
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
> Retrieval works with **no API key and no external inference server**. Required enrichment runs through managed
> Qwen3.5 0.8B ONNX on CPU. Synthesis is optional through any OpenAI-compatible endpoint — local
> ([vLLM](https://github.com/vllm-project/vllm), LM Studio, Ollama)
> or cloud (OpenAI, Together, Groq, Fireworks, OpenRouter, …).

```bash
pip install fitz-sage

# Start with governed evidence. From a docs folder, --source is optional.
fitz query "What is our refund policy?" --source ./docs

# Optional: synthesize an answer from that evidence
fitz answer "What is our refund policy?" --source ./docs \
  --endpoint http://localhost:8080/v1 \
  --synthesizer endpoint/gpt-oss-20b
```

That's it. Your documents are now searchable with governed provenance first.
`fitz query` returns evidence, not a generated answer.


![fitz-sage quickstart demo](https://raw.githubusercontent.com/yafitzdev/fitz-sage/main/docs/assets/quickstart_demo.gif)
*Figure 1: Example of user experience for querying documents using fitz-sage.*

---

### About

Existing RAG tools hallucinate. When the answer isn't in your documents, they invent one — confidently, fluently, wrongly. 
In production, that's not a minor inconvenience. It's the reason you can't trust the system. I built fitz-sage to solve that 
problem directly, while working as a Data Engineer in the automotive industry. No LangChain. No LlamaIndex. Every layer written from scratch.

The retrieval architecture is [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md) — documents are parsed into typed units (
code symbols, sections, tables) and each query is routed to the right search strategy, rather than searching flat chunks uniformly.

Honesty is enforced by [**pyrrho**](https://huggingface.co/yafitzdev/pyrrho-nano-g3) — a fine-tuned ModernBERT encoder that classifies every `(query, retrieved contexts)` pair into
`TRUSTWORTHY` / `DISPUTED` / `ABSTAIN` in a local INT8 ONNX forward pass on CPU. No LLM dependency on the governance path.
Validated against [fitz-gov](https://github.com/yafitzdev/fitz-gov), a purpose-built benchmark of 24,592 adversarial evidence-governance cases:
**97.52% overall accuracy** and **1.42% false-trustworthy rate**.

It runs in production today and powers [fitz-forge](https://github.com/yafitzdev/fitz-forge).

Yan Fitzner — ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev), [HuggingFace](https://huggingface.co/yafitzdev)).

![fitz-sage honest_rag](https://raw.githubusercontent.com/yafitzdev/fitz-sage/main/docs/assets/honest_rag.jpg)

---

<details>

<summary><strong>📦 What is RAG?</strong></summary>

<br>

RAG is how ChatGPT's "file search," Notion AI, and enterprise knowledge tools actually work under the hood.
Instead of sending all your documents to an AI, RAG:

1. [X] **Indexes your documents** — Stores searchable source units in a database
2. [X] **Retrieves only what's relevant** — When you ask a question, finds the best source units
3. [X] **Optionally sends those sources to an LLM** — Generated answers are a layer on top of retrieval

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
> GPT-5 accepts ~272k tokens. That's roughly 600 pages. Your company wiki, codebase, or document archive is likely 5x-50x 
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
> Ask a question immediately — no ingestion command required. `fitz-sage` parses a searchable surface, returns governed
> evidence, and then lets a background daemon finish managed Qwen keyword/entity/hierarchy enrichment.

**Honest answers ✅** → [pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-nano-g3)
> Most RAG tools confidently answer even when the answer isn't in your documents. Ask "What was our Q4 revenue?" when
> your docs only cover Q1-Q3, and typical RAG hallucinates a number. `fitz-sage` says: *"I cannot find Q4 revenue figures
> in the provided documents."*
>
> → Detects when to abstain at **97.83% recall** on [fitz-gov](https://github.com/yafitzdev/fitz-gov), a 24,592-case benchmark for
> evidence governance.
> Overall accuracy: **97.52%**. False-trustworthy rate: **1.42%**. One local encoder forward pass, no LLM call.

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
> stores tables natively in SQLite alongside every other retrieval unit—one `.db` per collection, no sync issues. Auto-detects 
> schema and runs real SQL. Ask "What's the average price by region?" and get an actual computed answer, not fragmented rows.

**Fully local execution possible 🏠** → [OpenAI-Compatible Endpoint](docs/features/platform/openai-compatible-endpoint.md)
> Embedded SQLite + managed ONNX enrichment. Optional synthesis can use any local OpenAI-compatible server (vLLM, LM Studio, Ollama) with no API key.

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
| [**Tables 📅**](docs/features/ingestion/tabular-data-routing.md) | CSV files or tables within documents | Native SQLite storage with auto-detected schema. Real SQL execution from natural language — not chunked text. |
| **Images 🖼️** | Figures and diagrams within documents | VLM-powered figure extraction and visual understanding. *(Coming soon)* |
| **Chunks 🧩** | Any content as fallback | Traditional chunk-based retrieval when structured extraction doesn't apply. Automatic fallback — no configuration needed. |

<br>

> [!NOTE]
> All retrieval units share the same retrieval intelligence (temporal handling, comparison queries, multi-hop reasoning, etc.) 
> and the same staged enrichment pipeline (query-ready keywords first; entities and hierarchy continue in the background).

---

### Retrieval Intelligence

Most RAG implementations are naive vector search — they fail silently on real-world queries.
`fitz-sage` runs retrieval as a **broad recall → rerank → Pyrrho cutoff** pipeline:

<br>

Full explanation: [Three-Stage Retrieval Strategy](docs/features/retrieval/three-stage-strategy.md).

<br>

| Stage | What it does | Main cost |
|-------|--------------|-----------|
| **1. Broad recall** | Extract real query terms, add semantic keywords, run BM25 over typed units, and fan out only when the query needs it (comparison, temporal, aggregation, multi-query). False positives are acceptable here. | Cheap SQLite FTS + deterministic query prep |
| **2. Rerank** | INT8 ONNX cross-encoder reorders the broad candidate list by true query relevance. This is where precision belongs. | Local ONNX reranker |
| **3. Pyrrho cutoff** | `pyrrho` evaluates `query + top 1`, then `query + top 2`, and so on until evidence is enough or the cutoff is reached. | Local ONNX classifier |

<br>

The quick path is keyword-first: real query keywords + Qwen semantic keywords + BM25. L1 summaries, entity graph links, and corpus hierarchy improve fully indexed retrieval, but they are not required before the first governed evidence pack can be returned.

Existing retrieval strategies are not deleted; they move into clear roles:

| Strategy | Role in the pipeline |
|----------|----------------------|
| Keyword vocabulary, sparse BM25 | Broad recall backbone |
| Query expansion | Broad recall keyword enrichment |
| Query rewriting | Transform only when conversational context or ambiguity would harm recall |
| Multi-query decomposition | Bounded broad recall for compound questions |
| Comparison, temporal, aggregation, freshness | Recall fanout and scoring hints |
| Hierarchical summaries | Fully indexed recall and rerank evidence, not the instant gate |
| Entity graph | Fully indexed context expansion after initial ranking |
| Reranking | Mandatory precision stage before governance |
| Multi-hop | Fallback bridge retrieval after the cutoff loop still abstains |
| Agentic manifest search | Fallback only for files that are not query-ready yet |

<br>

Across those tiers, [built-in intelligence](docs/features/retrieval) handles the edge cases that break naive RAG:

<br>

| Feature | Query | Naive RAG Problem | `fitz-sage` Solution |
|---------|-------|-------------------|------------------|
| [**epistemic-honesty**](docs/features/governance/epistemic-honesty.md) | "What was our Q4 revenue?" | ❌ Hallucinated number — Info doesn't exist, but LLM won't admit it | ✅ "I don't know" |
| [**keyword-vocabulary**](docs/features/retrieval/keyword-vocabulary.md) | "Find TC_1000" | ❌ Wrong test case — Embeddings see TC_1000 ≈ TC_2000 (semantically similar) | ✅ Exact keyword matching |
| [**sparse-search**](docs/features/retrieval/sparse-search.md) | "error code E_AUTH_401" | ❌ No exact match — Embeddings miss precise error codes | ✅ SQLite FTS5 + native `bm25()` |
| [**multi-hop**](docs/features/retrieval/multi-hop-reasoning.md) | "Who wrote the paper cited by the 2023 review?" | ❌ Returns the review only — Single-step search can't traverse references | ✅ Iterative retrieval |
| [**hierarchical-rag**](docs/features/ingestion/hierarchical-rag.md) | "What are the design principles?" | ❌ Random fragments — Answer is spread across docs; no single chunk contains it | ✅ Hierarchical summaries |
| [**multi-query**](docs/features/retrieval/multi-query-rag.md) | *[User pastes 500-char test report]* "What failed and why?" | ❌ Vaguely related chunks — Long input gets averaged, matches nothing specifically | ✅ Multi-query decomposition |
| [**comparison-queries**](docs/features/retrieval/comparison-queries.md) | "Compare React vs Vue performance" | ❌ Incomplete comparison — Only retrieves one entity, missing the other | ✅ Multi-entity retrieval |
| [**entity-graph**](docs/features/retrieval/entity-graph.md) | "What else mentions AuthService?" | ❌ Isolated chunks — No awareness of shared entities across docs | ✅ Entity-based linking across sources |
| [**temporal-queries**](docs/features/retrieval/temporal-queries.md) | "What changed between Q1 and Q2?" | ❌ Random chunks — No awareness of time periods in query | ✅ Temporal query handling |
| [**aggregation-queries**](docs/features/retrieval/aggregation-queries.md) | "List all the test cases that failed" | ❌ Partial list — No mechanism for comprehensive retrieval | ✅ Aggregation query handling |
| [**freshness-authority**](docs/features/retrieval/freshness-authority.md) | "What's the latest status on feature X?" | ❌ Old docs rank equally — No awareness of how recent a document is | ✅ Recency boosting |
| [**query-expansion**](docs/features/retrieval/query-expansion.md) | "How do I fetch the db config?" | ❌ No matches — User says "fetch", docs say "retrieve"; "db" vs "database" | ✅ Dictionary + managed-Qwen keyword expansion (no embeddings) |
| [**query-rewriting**](docs/features/retrieval/query-rewriting.md) | "Tell me more about it" *(after discussing TechCorp)* | ❌ Lost context — Pronouns like "it" reference nothing, retrieval fails | ✅ Conversational context resolution |
| [**reranking**](docs/features/retrieval/reranking.md) | "What's the battery warranty?" | ❌ Imprecise ranking — BM25 ≠ true relevance; best answer buried | ✅ ONNX cross-encoder reranker, ~30 ms on CPU |

<br>

> [!IMPORTANT]
> These features are **always on**—no configuration needed. `fitz-sage` automatically detects when to use each capability.

---

### Governance — Know What You Don't Know

[Feature docs](docs/CONSTRAINTS.md) • [pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-nano-g3) • [fitz-gov benchmark](https://github.com/yafitzdev/fitz-gov)

Most RAG systems hallucinate confidently. `fitz-sage` **measures and enforces** epistemic honesty using
[**pyrrho**](https://huggingface.co/yafitzdev/pyrrho-nano-g3) — a fine-tuned ModernBERT-base encoder served as
INT8 ONNX. One local forward pass per query, no external LLM call.

<br>

```
  Query + Retrieved Contexts
               │
               ▼
  ┌──────────────────────────┐
  │  pyrrho (ModernBERT,     │   single INT8 ONNX forward pass
  │  INT8 ONNX, local CPU)   │   split ONNX export
  └────────────┬─────────────┘
               │ softmax → (p_abstain, p_disputed, p_trustworthy)
               ▼
  Calibrated threshold (TAU = 0.60 on P(TRUSTWORTHY))
               │
               ▼
  TRUSTWORTHY  /  DISPUTED  /  ABSTAIN  →  EvidencePack / optional synthesis
```

<br>

| Decision        | Meaning                              | Recall    |
|-----------------|--------------------------------------|-----------|
| **ABSTAIN**     | Evidence doesn't answer the question | **97.83%** |
| **DISPUTED**    | Sources contradict each other        | **98.34%** |
| **TRUSTWORTHY** | Consistent, sufficient evidence      | **96.28%** |

**Overall accuracy: 97.52% ± 0.43** | **False-trustworthy: 1.42% ± 0.16** on fitz-gov V8.0.0 (3-seed mean, 2,459-case held-out test split)

<br>

> [!NOTE]
> Governance asks "given three relevant documents that partially contradict each other, should you flag a dispute, hedge
> the answer, or trust the consensus?" That's a judgment call even humans disagree on. Pyrrho was trained on 2,920 labeled
> cases from fitz-gov V8.0.0 to make those calls reproducibly.

<strong>The system fails safe 🛡️</strong>
> Threshold calibration is tuned on the `TRUSTWORTHY` probability: when pyrrho is uncertain, it falls back to the runner-up
> between `ABSTAIN` and `DISPUTED`. Over-confidence is the rarest error mode.

<strong>No LLM on the governance path ⏱️</strong>
> Pyrrho replaces a 5-call constraint cascade with a single encoder forward pass — ~50× faster, zero external API
> dependency for governance, and +7.43 pp more accurate than the cascade it replaced.

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
>`fitz-sage` creates a retrieval-first config on first run:
>1. **Local OpenAI-compatible server running?** → Uses it automatically (probes ports 8080 / 8000 / 1234 / 11434 for `/v1/models`)
>2. **`OPENAI_API_KEY` set?** → Uses it automatically
>3. **Neither?** → Retrieval still works; synthesis stays disabled until you configure a provider
>
>For one-off synthesized answers against any OpenAI-compatible URL, skip the config:
>
>```bash
>fitz answer "..." --endpoint http://localhost:8080/v1 \
>                 --synthesizer endpoint/gpt-oss-20b
>fitz answer "..." --endpoint https://api.together.xyz/v1 \
>                 --synthesizer endpoint/meta-llama-3.1-70b \
>                 --api-key-env TOGETHER_API_KEY
>```

<br>

#### Python SDK
>
>```python
>import fitz_sage
>
>pack = fitz_sage.evidence("Your question here", source="./docs")
>
>print(pack.mode)
>for item in pack.items:
>    print(f"  - {item.file_path}: {item.excerpt[:50]}...")
>```
>
>The SDK provides:
>- Module-level `evidence()` matching `fitz query`
>- Module-level `query()` for optional synthesis
>- Retrieval-only auto-config creation
>- Full provenance tracking
>- Same honest retrieval as the CLI
>
>For advanced use (multiple collections), use the `fitz` class directly:
>```python
>from fitz_sage import fitz
>
>physics = fitz(collection="physics")
>pack = physics.evidence("Explain entanglement", source="./physics_papers")
>```

<br>

#### Fully Local (Managed ONNX)
>
>```bash
>pip install fitz-sage
>
>fitz query "Your question here" --source ./docs
>```
>
>Reranking, governance, and required enrichment run as local ONNX models on CPU — no separate embedding server, no second API key.
>No data leaves your machine.
>
>Other compatible servers: [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.ai) 
> (in `/v1/` mode), [TabbyAPI](https://github.com/theroyallab/tabbyAPI). Anything that speaks the OpenAI HTTP protocol works.

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

> **Code retrieval:**
>
> tree-sitter parses your codebase into symbols (functions, classes, methods) with qualified names, 
> references, and import graphs. No chunking—each symbol is a precise, addressable unit. Cross-file dependencies are 
> tracked, so "what calls this function?" is a graph traversal, not a text search.
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
│                         fitz-sage                             │
├───────────────────────────────────────────────────────────────┤
│  User Interfaces                                              │
│  CLI: query | retrieve | answer | collections | serve         │
│  SDK: fitz_sage.evidence(source=...)                          │
│  API: /query | /chat | /collections | /health                 │
├───────────────────────────────────────────────────────────────┤
│  Engines                                                      │
│  ┌────────────┐  ┌────────────┐                               │
│  │  FitzKRAG  │  │  Custom... │  (extensible registry)        │
│  └────────────┘  └────────────┘                               │
├───────────────────────────────────────────────────────────────┤
│  Optional LLM Provider (single OpenAI-compatible HTTP protocol)│
│  synthesis | query-intelligence | vision                      │
├───────────────────────────────────────────────────────────────┤
│  Local CPU encoders (INT8 ONNX, no external calls)            │
│  pyrrho (governance)  |  gte-reranker-modernbert-base         │
├───────────────────────────────────────────────────────────────┤
│  Storage (SQLite + FTS5, one .db per collection)              │
│  symbols | sections | tables | full-text search (bm25)        │
├───────────────────────────────────────────────────────────────┤
│  Retrieval (address-based, baked-in intelligence)             │
│  symbols | sections | tables | import graphs | reranking      │
├───────────────────────────────────────────────────────────────┤
│  Managed Qwen enrichment (required, local ONNX)               │
│  semantic query keywords | entities | hierarchy | summaries   │
├───────────────────────────────────────────────────────────────┤
│  Governance (epistemic safety)                                │
│  pyrrho encoder | TRUSTWORTHY / DISPUTED / ABSTAIN, local CPU │
└───────────────────────────────────────────────────────────────┘
```

</details>

---

<details>

<summary><strong>📦 CLI Reference</strong> → <a href="docs/CLI.md">Full CLI Guide</a></summary>

<br>

```bash
fitz query "question" --source ./docs     # Point at docs and retrieve evidence
fitz query "question"                     # Use current folder or existing collection
fitz retrieve "question" --format json    # Evidence with script-friendly controls
fitz answer "question" --synthesizer ...  # Optional synthesized answer
fitz collections                       # List and delete knowledge collections
fitz serve                             # Start REST API server
```

Config: `~/.fitz/config/fitz_krag.yaml` — auto-created on first run. Retrieval works with managed local ONNX models; edit config only for optional synthesis, query intelligence, or vision.

</details>

---

<details>

<summary><strong>📦 Python SDK Reference</strong> → <a href="docs/SDK.md">Full SDK Guide</a></summary>

<br>

**Simple usage (module-level, matches CLI):**
```python
import fitz_sage

pack = fitz_sage.evidence("What is the refund policy?", source="./docs")
print(pack.mode)
```

<br>

**Advanced usage (multiple collections):**
```python
from fitz_sage import fitz

# Create separate instances for different collections
physics = fitz(collection="physics")
legal = fitz(collection="legal")

# Retrieve evidence from each collection
physics_pack = physics.evidence("Explain entanglement", source="./physics_papers")
legal_pack = legal.evidence("What are the payment terms?", source="./contracts")
```

<br>

**Working with evidence:**
```python
pack = fitz_sage.evidence("What is the refund policy?")

print(pack.mode)  # TRUSTWORTHY, DISPUTED, or ABSTAIN

for item in pack.items:
    print(f"Source: {item.source_id}")
    print(f"Excerpt: {item.excerpt}")
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
> This only applies to optional endpoint-backed synthesis or query intelligence. Use `fitz query "..."` for evidence without an endpoint server. For a one-off synthesized answer:
> `fitz answer "..." --synthesizer openai/gpt-4o`.

**"Model not found" error**
> The model name in your config doesn't match what your server has loaded. Check `/v1/models` on your server:
> `curl http://localhost:8080/v1/models`. Then update `synthesizer` in `~/.fitz/config/fitz_krag.yaml` to match.

**First query is slow**
> First run initializes the database, downloads local ONNX models if needed, and indexes source files. Subsequent retrievals
> are much faster. Optional synthesis can still be slow if a local chat model is cold.

**How do I change my LLM endpoint or model?**
> Edit `~/.fitz/config/fitz_krag.yaml`:
> ```yaml
> synthesizer: endpoint/gpt-oss-20b
> chat_base_url: http://localhost:8080/v1
> ```
> Or override at the CLI without editing YAML:
> ```bash
> fitz answer "..." --endpoint http://localhost:8080/v1 --synthesizer endpoint/gpt-oss-20b
> ```

**How do I use a cloud provider?**
> Either use the `openai` preset (built-in OpenAI URL):
> ```yaml
> synthesizer: openai/gpt-4o
> # OPENAI_API_KEY in env
> ```
> Or any OpenAI-compatible cloud via the `endpoint` provider:
> ```yaml
> synthesizer: endpoint/meta-llama-3.1-70b
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
- [Unified Storage (SQLite + FTS5)](docs/features/platform/unified-storage.md)
- [Progressive KRAG & Agentic Search](docs/features/platform/progressive-krag-agentic-search.md)
- [Ingestion Pipeline](docs/INGESTION.md)
- [Enrichment (Hierarchies, Entities)](docs/ENRICHMENT.md)
- [Epistemic Governance (pyrrho)](docs/CONSTRAINTS.md)
- [Governance Benchmarking (fitz-gov)](docs/features/governance/governance-benchmarking.md)
- [BEIR Benchmark Results](docs/evaluation/beir-results.md)
- [Plugin Development](docs/PLUGINS.md)
- [Feature Control](docs/FEATURE_CONTROL.md)
- [KRAG — Knowledge Routing Augmented Generation](docs/features/platform/krag.md)
- [Code Symbol Extraction](docs/features/ingestion/code-symbol-extraction.md)
- [Tabular Data Routing](docs/features/ingestion/tabular-data-routing.md)
- [Enterprise Gateway](docs/features/platform/enterprise-gateway.md)
- [Engines](docs/ENGINES.md)
- [Configuration Examples](docs/CONFIG_EXAMPLES.md)
- [Custom Engines](docs/CUSTOM_ENGINES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
