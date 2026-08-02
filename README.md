<!-- README.md -->

<div align="center">

# fitz-sage

### Fully local governed RAG for code, documents, and tables.

**No LLM API key or GPU required.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-sage.svg)](https://pypi.org/project/fitz-sage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.16.0-green.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/yafitzdev/fitz-sage)

[Benchmarks](#benchmarks) • [EvidencePack](#evidencepack) • [Why `fitz-sage`?](#why-fitz-sage) • [Retrieval Intelligence](#retrieval-intelligence) • [Governance](#governance--pyrrho) • [Limitations](#limitations) • [Documentation](#links) • [GitHub](https://github.com/yafitzdev/fitz-sage)

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
    to answer this question."
</pre><pre>
    Related topics in the knowledge base:
      - FIFA tournament history
      - 2022 World Cup coverage
</pre><pre>
    To answer this, consider adding:
      - Documents covering 2024 FIFA events.
</pre>
    </td>
  </tr>
</table>

  → `fitz-sage` returns governed evidence, explains insufficiency, and shows what source coverage is missing.
</div>

---

### Where to start 🚀

> [!IMPORTANT]
> `fitz retrieve` runs locally by default. SQLite stores the index; local models
> handle semantic query terms, reranking, Pyrrho governance, and optional
> background enrichment. An OpenAI-compatible endpoint is only needed for an
> explicitly configured endpoint-backed role such as generated prose.

```bash
pip install fitz-sage

# From a docs folder, --source is optional.
fitz retrieve "What is our refund policy?" --source ./docs
```

The result is an `EvidencePack`: relevant source units, provenance, a governance verdict, and the signals needed to decide
what your application should do next.

---

### About

`fitz-sage` is a retrieval engine for local knowledge bases. It indexes code, documents, and tables into typed source units,
retrieves the units relevant to a question, reranks them, and returns a governed `EvidencePack` that downstream software can
inspect, display, store, or pass to a synthesizer.

⭐ The retrieval architecture is [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md). Code is parsed
as symbols, documents as sections, and tables as SQLite-backed data. Queries are routed across those
typed surfaces with retrieval strategies that match the source structure.

⭐ Governance is enforced by [Pyrrho](https://huggingface.co/yafitzdev) in local CPU forward passes. Fitz starts with the
first three ranked sources and adds two only while Pyrrho returns `INSUFFICIENT`.

Yan Fitzner — ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev), [HuggingFace](https://huggingface.co/yafitzdev)).

![fitz-sage honest_rag](https://raw.githubusercontent.com/yafitzdev/fitz-sage/main/docs/assets/honest_rag.jpg)

---

### Why `fitz-sage`?

**EvidencePack as the contract 🧾**
> Every query returns ranked source evidence, provenance, governance reasons, and retrieval metadata. Use it directly in
> APIs, CLIs, dashboards, agents, or pass it to an LLM for answer generation with explicit governance metadata.

**Asymmetric indexing 🗂️** → [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md)
> Source files become typed retrieval units: code symbols, document sections, and tables. Each unit type
> keeps the structure needed to retrieve it well.

**Query-ready indexing 🐆** → [Searchable Index](docs/features/platform/searchable-index-background-enrichment.md)
> `point()` parses and stores supported files before returning. Retrieval can start immediately afterward while the
> background worker adds optional entity and hierarchy metadata.

**Pyrrho-governed retrieval 🧭** → [Pyrrho docs](docs/CONSTRAINTS.md)
> Fitz combines deterministic query shape with Pyrrho PRE obligations before retrieval, then Pyrrho judges the selected evidence after reranking. The retrieval profile,
> reasons, and missing-evidence signals travel with the `EvidencePack`, so callers know whether to answer, show conflict,
> retrieve more, or ask for more source material.

**Queries that actually work 📊**
> Exact identifiers, temporal scopes, comparisons, aggregation requests, code lookups, table questions, and broad overview
> queries all flow through retrieval intelligence built into the engine.

**Tabular data that is actually searchable 📈** → [Unified Storage](docs/features/platform/unified-storage.md)
> Native CSV/TSV rows live in SQLite with schema detection, row-value BM25, and
> deterministic row grounding. Optional configured chat tiers can generate SQL
> for one retrieved table. Embedded document tables remain section text.

**Fully local execution possible 🏠**
> SQLite storage, ONNX reranking, managed Qwen query/background work, and ONNX Pyrrho governance all run locally. Optional synthesis can use
> any local or cloud OpenAI-compatible endpoint.

####

> [!TIP]
> Try fitz on itself:
>
> ```bash
> fitz retrieve "How does the retrieval pipeline work?" --source ./fitz_sage
> ```

---

### What You Can Search

Traditional documents, source code, and tables have different structure. FitzKRAG preserves that structure during indexing
and retrieval.

<br>

| Retrieval Unit | Extracted From | How It Works |
|----------------|----------------|--------------|
| [**Symbols 🖌️**](docs/features/ingestion/code-symbol-extraction.md) | Python, TypeScript/JavaScript, Go, Java | AST/tree-sitter strategies extract addressable names, ranges, references, and file import edges. |
| **Sections 📑** | PDF, DOCX, PPTX, Markdown, text, config, markup | Parsed source text becomes sections with headings, ranges, and parent/child context; summaries are optional background metadata. |
| [**Tables 📅**](docs/features/ingestion/tabular-data-routing.md) | Configured delimited files (`.csv`, `.tsv` by default) | Native SQLite rows with schema lookup, row-value BM25, deterministic grounding, and optional generated SQL. |

<br>

> [!NOTE]
> All retrieval units share the same retrieval intelligence: query profiling, temporal handling, comparisons,
> aggregation, keyword expansion, reranking, progressive evidence delivery, and Pyrrho governance.

---

### Retrieval Intelligence

[Retrieval Docs](docs/features/retrieval/README.md) • [Three-Stage Strategy](docs/features/retrieval/three-stage-strategy.md) • [Retrieval Pipeline](docs/RETRIEVAL_PIPELINE.md) • [Evidence Signals](docs/features/retrieval/evidence-signals.md)

`fitz-sage` runs retrieval as a typed, governed pipeline:

<br>

| Stage | What happens |
|-------|--------------|
| **1. Broad recall 🔎** | Finds candidate evidence:<br>`Doc 2`<br>`Doc 3`<br>`Doc 5`<br>`Doc 8` |
| **2. Rerank 🎯** | Reorders by relevance:<br>`Doc 5`<br>`Doc 2`<br>`Doc 8`<br>`Doc 3` |
| **3. Governance 🛡️** | Grows the ranked prefix until Pyrrho decides:<br>`Doc 5 + Doc 2 + Doc 8` → `INSUFFICIENT`<br>`+ Doc 3` → `SUFFICIENT` |

<br>

The query-ready path is keyword-first: exact query terms, Qwen semantic
keywords, and BM25. Enriched collections can additionally use hierarchy
summaries, entity links, and broader context expansion.

[Built-in intelligence](docs/features/retrieval) handles the edge cases that break simple search:

<br>

| Feature | Query | What Fitz Uses |
|---------|-------|----------------|
| ✅ [**epistemic-honesty**](docs/features/governance/epistemic-honesty.md) | "What was our Q4 revenue?" | Pyrrho verdict and insufficient-evidence reasons |
| ✅ [**keyword-vocabulary**](docs/features/retrieval/keyword-vocabulary.md) | "Find TC_1000" | Literal identifier search |
| ✅ [**sparse-search**](docs/features/retrieval/sparse-search.md) | "error code E_AUTH_401" | SQLite FTS5 + native `bm25()` |
| ✅ [**hierarchical-rag**](docs/features/ingestion/hierarchical-rag.md) | "What are the design principles?" | Hierarchical summaries when enrichment has produced them |
| ✅ [**multi-query**](docs/features/retrieval/multi-query-rag.md) | *[User pastes 500-char test report]* "What failed and why?" | Multi-query decomposition |
| ✅ [**comparison-queries**](docs/features/retrieval/comparison-queries.md) | "Compare React vs Vue performance" | Multi-entity retrieval coverage |
| ✅ [**entity-graph**](docs/features/retrieval/entity-graph.md) | "What else mentions AuthService?" | Entity links for enriched source files |
| ✅ [**temporal-queries**](docs/features/retrieval/temporal-queries.md) | "What changed between Q1 and Q2?" | Temporal scope detection |
| ✅ [**aggregation-queries**](docs/features/retrieval/aggregation-queries.md) | "List all the test cases that failed" | Exhaustive/list query handling |
| ✅ [**freshness-authority**](docs/features/retrieval/freshness-authority.md) | "What's the latest status on feature X?" | Content-grounded temporal scope; no filesystem-age scoring |
| ✅ [**semantic-keywords**](docs/features/retrieval/query-expansion.md) | "How do I fetch the db config?" | Managed-Qwen recall terms merged with literal query terms |
| ✅ [**query-rewriting**](docs/features/retrieval/query-rewriting.md) | "Tell me more about it" *(after discussing TechCorp)* | Configured query-intelligence provider plus caller-supplied history |
| ✅ [**reranking**](docs/features/retrieval/reranking.md) | "What's the battery warranty?" | ONNX cross-encoder reranker |

<br>

> [!IMPORTANT]
> Retrieval intelligence is baked in. Configuration declares providers; the engine decides which retrieval capabilities a
> query needs.

---

<a id="benchmarks"></a>

<details>

<summary><strong>📦 Benchmarks</strong></summary>

<br>

[Full Benchmark Report](docs/BENCHMARK.md) • [Reproduce the Benchmarks](benchmarks/README.md) • [Production Readiness](docs/PRODUCTION_READINESS.md) • [Measured Limitations](docs/LIMITATIONS.md)

Benchmarks start with source files and report retrieval, delivery, ingestion,
and latency separately.

| Area | Scale | Current measurement |
|------|------:|---------------------|
| Production retrieval and delivery | 192 required contracts | 190/192 compiled; 172/192 delivered |
| Query-shape recognition | 60 cases | 60/60 |
| Intentional limitations | 52 evidence-asserted cases | 51/52 compiled; 48/52 delivered |
| Broad BEIR | 66,454 documents, 1,271 queries | 0.4239 delivered nDCG@10 |
| Frozen semantic BEIR | 531,605 documents, 240 queries | 0.6519 delivered nDCG@10 |
| EnterpriseRAG-Bench | 511,961 documents, 328 holdout queries | 0.5780 delivered nDCG@10 |
| Local source indexing | 18 core / 93 mixed files | 60.8 / 51.6 files/s |
| NapierOne scale ingestion | 5,005 real files | 4,994 indexed at 7.27 files/s; recovery passed |
| SciFact query latency | 60 matched queries | 7.43s mean; 6.77s p50; 12.56s p95 |
| Enterprise warm query probes | 511,961-file index | 13.092s and 19.889s |

> nDCG@10 measures ranking quality in the first ten results; it is not an
> accuracy percentage. Full methodology, component results, timings, and
> interpretation are in the [benchmark report](docs/BENCHMARK.md).

</details>

---

<a id="evidencepack"></a>

<details>

<summary><strong>📦 EvidencePack</strong> → <a href="docs/EVIDENCE_PACK.md">Full Contract</a></summary>

<br>

[Evidence Pack Contract](docs/EVIDENCE_PACK.md) • [Evidence Signals](docs/features/retrieval/evidence-signals.md)

`EvidencePack` is the output contract of `fitz-sage`.

It gives you the relevant sources and the governance signals around them. You can show it directly, pass it to a model,
trigger a workflow from it, or store it as an audit artifact.

The source items are the evidence. The signals around them explain how Fitz searched before retrieval and what Pyrrho judged
after retrieval.

#### Pre-retrieval 🔎

Before retrieval, Fitz builds a search plan from deterministic query analysis,
managed Qwen query keywords, and optional query intelligence.

| Signal | What it means | Why it matters |
|--------|---------------|----------------|
| `query_type` / `analysis_type` | Narrow lookup, comparison, temporal, aggregation, broad overview, or general query shape. | Sets recall breadth and evidence coverage. |
| `keywords` | Managed Qwen suggestions and literal deterministic query terms. | Adds best-effort lexical candidates without embeddings. |
| `strategy_weights` | Relative weight for code, section, and table retrieval. | Makes the first pass search the right evidence surfaces. |
| `top_k` / `top_read` | How much candidate evidence Fitz should collect and read. | Keeps narrow lookups fast while giving broad or comparative questions enough coverage. |
| `rerank_candidates` | How many recalled candidates the cross-encoder scores. | Bounds neural CPU cost without shrinking the BM25 recall pool used by evidence rescue. |

#### Post-retrieval 🛡️

After retrieval, reranking, closure, and compilation, Fitz sends Pyrrho the
first three ranked items. An exact `INSUFFICIENT` verdict adds the next two;
`SUFFICIENT` or `DISPUTED` stops immediately. These signals tell you whether
the result is usable.

| Signal | What it means | What you can do with it |
|--------|---------------|-------------------------|
| `mode` | Mechanical Fitz-Sage mapping of Pyrrho's `SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT` verdict. | Gate generated answers, UI display, automation, or human review. |
| `reasons` | Plain-language explanation for the verdict. | Show users why Fitz judged evidence sufficient, disputed, or insufficient. |
| `evidence_verdict` | Verdict: `SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT`. | Inspect the evidence judgment. |
| `failure_mode` | Reason when evidence is insufficient or disputed. | Explain why the evidence cannot safely support a clean answer. |
| `retrieval_intents` | Evidence intent metadata such as lookup, temporal resolution, comparison, or broad coverage. | Decide whether another retrieval pass should focus on coverage, time, lookup, or comparison. |
| `evidence_kinds` | Evidence-surface metadata such as text, table, code, config, logs, or document layout. | Decide which evidence surface is missing or should be emphasized. |

This is why `fitz-sage` is useful as infrastructure: the package returns source evidence plus enough judgment to decide the
next action.

#### Retrieval execution records

When an `EvidencePack` is not enough to diagnose a result, `RetrievalRun`
records the actual query plan, term origins, candidate stages, compiled ranking,
evaluated evidence prefixes, exact Pyrrho outputs, and runtime fingerprints
from the same execution.

```bash
fitz retrieve "Which test failed?" -c reports --trace run.json
fitz explain run.json
```

Trace exports redact source bodies by default. Content-bearing traces are an
explicit opt-in and enable Pyrrho-only replay over frozen evidence. See
[Retrieval Execution Records](docs/RETRIEVAL_RUNS.md).

</details>

---

<a id="governance--pyrrho"></a>

<details>

<summary><strong>📦 Governance — Pyrrho</strong> → <a href="docs/CONSTRAINTS.md">Feature Docs</a></summary>

<br>

[Feature docs](docs/CONSTRAINTS.md) • [Pyrrho on Hugging Face](https://huggingface.co/yafitzdev) • [fitz-gov on Hugging Face](https://huggingface.co/datasets/yafitzdev/fitz-gov-v2)

Pyrrho is the local governance model behind `fitz-sage`. Its default CPU-local
ONNX ModernBERT model
[`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1)
is pinned to an immutable Hub revision and cached by Fitz-Sage through the
standard Hugging Face cache.

<br>

```
  Query
    │
    ▼
  RetrievalProfile → broad recall → rerank
    │
    ▼
  Ranked evidence prefix (first 3)
    │
    ▼
  Pyrrho authoritative decision
    ├── SUFFICIENT / DISPUTED ───────────────────────→ EvidencePack
    ├── INSUFFICIENT + exhausted ────────────────────→ EvidencePack
    └── INSUFFICIENT + evidence remains → add next 2 ──┐
                ▲                                      │
                └──────────────────────────────────────┘
```

<br>

| Signal | Purpose |
|--------|---------|
| `evidence_verdict` | Evidence judgment: `SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT`. |
| `failure_mode` | Reason when evidence is insufficient or disputed. |
| `retrieval_intents` | Evidence intent metadata, such as lookup, temporal resolution, comparison, or broad coverage. |
| `evidence_kinds` | Evidence-surface metadata, such as text, table, code, config, logs, or document layout. |

Fitz-Sage passes every ranked prefix to Pyrrho unchanged. Its managed ONNX
adapter applies the model's fixed input and head-decoding contract, maps the
resulting verdict into `AnswerMode`, and returns the stopping prefix plus the
exact serialized decisions with the `EvidencePack`. Applications can answer,
retry, show conflict, or request more source material.

<br>

> [!NOTE]
> Governance is a source-evidence judgment. Pyrrho is trained to decide whether retrieved evidence is sufficient,
> disputed, or insufficient, and Fitz records that judgment in the returned metadata.

<strong>The model adapter fails closed on known contract violations 🛡️</strong>
> Fitz-Sage's managed Pyrrho adapter checks the model artifact, label order, ONNX
> width, token limits, graph parity, and verdict/failure compatibility before
> or during inference. These
> checks reduce unsafe failure modes; they are not a substitute for clean-data
> evaluation or threshold calibration.

<strong>No LLM on the governance path ⏱️</strong>
> Pyrrho is a local encoder forward pass. Governance does not require an external chat model.

</details>

---

<a id="limitations"></a>

<details>

<summary><strong>📦 Limitations</strong> → <a href="docs/LIMITATIONS.md">Full Measured Contract</a></summary>

<br>

Fitz-Sage targets reasonably clean, supported documents. The complete contract
and case-level evidence are in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

| Boundary | Current behavior | Responsibility |
|----------|------------------|----------------|
| Identifier variants | `ATX-123`, `ATX_123`, and `ATX 123` remain distinct | User data preparation |
| Private vocabulary | Managed semantic terms are best effort; private mappings are not inferred | User data preparation |
| Raw logs and scans | Logs need compression; scans need an OCR/vision parser | User input pipeline |
| Long or unrelated requests | Retrieval and evidence budgets are finite | Shared boundary |
| Multi-document ranking | Weaker than single-document ranking on the enterprise holdout | Fitz-Sage |
| Extreme file counts | Public re-pointing still walks and hashes every source file | Fitz-Sage |
| Governance context | Pyrrho currently accepts up to 2,048 tokens | Pyrrho |

</details>

---

<details>

<summary><strong>📦 Quick Start</strong></summary>

<br>

> `governance: pyrrho` uses the accepted immutable default. Advanced users may
> instead configure `pyrrho/<absolute-local-path>` or an explicitly pinned
> `pyrrho/<owner/repo@40-character-commit>`.

#### CLI
>
>```bash
>pip install fitz-sage
>
>fitz retrieve "Your question here" --source ./docs
>```
>
>`fitz-sage` creates a local retrieval config on first run:
>1. **SQLite storage** for collections.
>2. **Managed local models** for semantic query terms, reranking, governance,
>   and optional background enrichment.
>3. **Pyrrho query planning** plus one authoritative evidence decision.
>
>For generated prose from the governed evidence:
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
>pack = fitz_sage.evidence("Where is Pyrrho governance implemented?", source="./fitz_sage")
>
>print(pack.mode)
>for item in pack.items:
>    print(item.file_path, item.address_location)
>```
>
>The SDK provides:
>- Module-level `evidence()` matching `fitz retrieve`
>- Module-level `answer()` for generated prose from evidence
>- Local config creation
>- Full provenance tracking
>- Governance metadata
>
>For advanced use with multiple collections:
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
>fitz retrieve "Your question here" --source ./docs
>```
>
>With the default retrieval-only config, reranking, governance, query-time
>expansion, and optional background enrichment run locally, so `fitz retrieve`
>does not send data to an endpoint. Explicitly configured query intelligence,
>chat tiers, or vision parsing can send query or source content to that endpoint.
>
>Optional synthesis can use [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai),
>[Ollama](https://ollama.ai) in `/v1/` mode, [TabbyAPI](https://github.com/theroyallab/tabbyAPI), OpenAI, Together,
>Groq, Fireworks, OpenRouter, or any endpoint that speaks the OpenAI HTTP protocol.

</details>

---

<details>

<summary><strong>📦 Real-World Usage</strong></summary>

<br>

`fitz-sage` is a retrieval foundation. It manages indexing, search, reranking,
Pyrrho integration, and provenance so products can
build on source evidence.

<br>

<strong>Chatbot Backend 🤖</strong>

> Connect fitz to Slack, Discord, Teams, or your own UI. The bot can show source-backed evidence, ask for more documents
> when Pyrrho marks evidence insufficient, or call `fitz answer` for generated prose.
>
> *Example:* A support bot retrieves policy sections, shows links to the relevant docs, and only synthesizes when the
> evidence verdict is sufficient.

<br>

<strong>Internal Knowledge Base 📖</strong>

> Point fitz at your wiki, policies, runbooks, and repos. Employees ask natural-language questions and get source units
> with provenance.
>
> *Example:* New hires ask "How do I request PTO?" and receive the exact policy section plus the governance verdict.

<br>

<strong>Continuous Intelligence & Alerting (Watchdog) 🐶</strong>

> Run scheduled queries over changing folders, logs, reports, or exports. Trigger alerts when the evidence pack contains
> sufficient sources, disputes, or missing-coverage signals.
>
> *Example:* A nightly job asks "Were there failed logins from unusual locations?" and sends the evidence pack to the
> on-call channel.

<br>

<strong>Web Knowledge Base 🌎</strong>

> Scrape web pages to disk, point fitz at the folder, and query the resulting corpus with provenance.
>
> *Example:* A research workflow scrapes reports, stores them locally, and asks comparative or temporal questions across
> the collected source set.

<br>

<strong>Codebase Search 🐍</strong> → [Code Symbol Extraction](docs/features/ingestion/code-symbol-extraction.md) • [KRAG](docs/features/platform/krag.md)

> **Code retrieval:**
>
> Tree-sitter parses your codebase into symbols with qualified names, references, and import graphs. Function and class
> lookup is address-based, and dependency questions can use graph expansion.
>
> *Example:* A team asks "Where is user authentication handled?" and receives specific functions, files, and symbol
> addresses rather than generic file snippets.

</details>

---

<details>

<summary><strong>📦 Architecture</strong> → <a href="docs/ARCHITECTURE.md">Full Architecture Guide</a></summary>

<br>

```
┌─────────────────────────────────────────────────────────────────┐
│                         fitz-sage                               │
├─────────────────────────────────────────────────────────────────┤
│  User Interfaces                                                │
│  CLI: retrieve | explain | replay | answer | collections | serve│
│  SDK: fitz_sage.evidence(source=...)                            │
│  API: /answer | /evidence | /chat | /collections | /health      │
├─────────────────────────────────────────────────────────────────┤
│  Engine                                                         │
│  FitzKRAG: typed retrieval over code, documents, and tables     │
├─────────────────────────────────────────────────────────────────┤
│  Evidence Contract                                              │
│  EvidencePack: items | mode | reasons | timings | metadata      │
├─────────────────────────────────────────────────────────────────┤
│  Pyrrho                                                         │
│  evidence verdict | failure mode | evidence metadata            │
├─────────────────────────────────────────────────────────────────┤
│  Local CPU Models                                               │
│  ONNX reranker | managed Qwen query/background | Pyrrho         │
├─────────────────────────────────────────────────────────────────┤
│  Storage                                                        │
│  SQLite + FTS5, one .db per collection                          │
├─────────────────────────────────────────────────────────────────┤
│  Optional OpenAI-Compatible Endpoint                            │
│  answer synthesis | query intelligence | vision                 │
└─────────────────────────────────────────────────────────────────┘
```

</details>

---

<details>

<summary><strong>📦 CLI Reference</strong> → <a href="docs/CLI.md">Full CLI Guide</a></summary>

<br>

```bash
fitz retrieve "question" --source ./docs     # Return governed evidence
fitz retrieve "question"                     # Use current folder or existing collection
fitz retrieve "question" --format json    # Evidence with script-friendly controls
fitz answer "question" --synthesizer ...  # Generated prose from evidence
fitz collections                          # List and delete knowledge collections
fitz serve                                # Start REST API server
```

Config: `.fitz/config.yaml` in the current workspace - auto-created on first run. Edit it for optional synthesis, query intelligence,
vision, or custom model/provider choices.

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
pack = fitz_sage.evidence("Where is Pyrrho governance implemented?", source="./fitz_sage")

print(pack.mode)  # runtime AnswerMode: SUFFICIENT, DISPUTED, or INSUFFICIENT
print(pack.reasons)

for item in pack.items:
    print(item.file_path, item.address_location, item.line_range)
```

</details>

---

<details>

<summary><strong>📦 REST API Reference</strong> → <a href="docs/API.md">Full API Guide</a></summary>

<br>

**Start the server:**
```bash
pip install fitz-sage[api]

# Initialize the workspace and collection once
fitz retrieve "What is indexed?" --source ./docs

fitz serve                    # localhost:8000
fitz serve -p 3000            # custom port
$env:FITZ_API_KEY = "replace-with-a-random-secret"
fitz serve --host 0.0.0.0     # remote access requires the API key
```

**Interactive docs:** Visit `http://localhost:8000/docs` for Swagger UI.

<br>

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/answer` | Return an optional synthesized answer |
| POST | `/evidence` | Return governed evidence without synthesis |
| POST | `/chat` | Return generated prose from retrieved evidence |
| GET | `/collections` | List all collections |
| GET | `/collections/{name}` | Get collection stats |
| POST | `/collections/{name}/documents` | Build/update the searchable source index |
| GET | `/collections/{name}/status` | Inspect source-index and enrichment status |
| DELETE | `/collections/{name}` | Delete a collection |
| GET | `/health` | Health check |

<br>

**Example request:**

`/answer` and `/chat` require a configured synthesizer. `/evidence` does not.

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "collection": "default"}'
```

</details>

---

<details>

<summary><strong>📦 FAQ / Troubleshooting</strong></summary>

<br>

**`fitz` command not found after install**
> Your Python Scripts directory is not on PATH. Use `python -m fitz_sage.cli.cli`, or add the Scripts directory to PATH.

**PDF/DOCX/PPTX files are being skipped**
> The base install reads embedded text from these formats. Image-only files
> need an OCR-capable parser. `parser: glm_ocr` expects a local Ollama
> `glm-ocr` model; install `fitz-sage[docs]` only when you explicitly select a
> Docling parser.

**"Connection refused at localhost:8080" error**
> This applies to optional endpoint-backed synthesis or query intelligence. `fitz retrieve "..."` returns evidence without an
> endpoint server. For generated prose:
> `fitz answer "..." --synthesizer openai/gpt-4o`.

**"Model not found" error**
> The model name in your config does not match what your server has loaded. Check `/v1/models` on your server:
> `curl http://localhost:8080/v1/models`. Then update `synthesizer` in `.fitz/config.yaml`.

**First query is slow**
> First run initializes storage, downloads managed local models if needed, and builds the query-ready index. Later queries
> reuse the collection.

**How do I change my LLM endpoint or model?**
> Edit `.fitz/config.yaml`:
> ```yaml
> synthesizer: endpoint/gpt-oss-20b
> chat_base_url: http://localhost:8080/v1
> ```
> Or override at the CLI:
> ```bash
> fitz answer "..." --endpoint http://localhost:8080/v1 --synthesizer endpoint/gpt-oss-20b
> ```

**How do I use a cloud provider?**
> Either use the `openai` preset:
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
> See [docs/features/platform/openai-compatible-endpoint.md](docs/features/platform/openai-compatible-endpoint.md).

**How do I reset everything?**
> Delete the `.fitz/` directory in your project root. Next run will initialize a fresh workspace.

</details>

---

### License

MIT

---

### Links

- [GitHub](https://github.com/yafitzdev/fitz-sage)
- [PyPI](https://pypi.org/project/fitz-sage/)
- [Changelog](CHANGELOG.md)
- [Benchmark Report](docs/BENCHMARK.md)
- [Benchmark Methodology](benchmarks/README.md)
- [Production Readiness](docs/PRODUCTION_READINESS.md)
- [Evaluation Reports](docs/evaluation/README.md)
- [Measured Limitations](docs/LIMITATIONS.md)

**Documentation:**
- [Docs Index](docs/README.md)
- [Evidence Pack Contract](docs/EVIDENCE_PACK.md)
- [Evidence Signals](docs/features/retrieval/evidence-signals.md)
- [Three-Stage Retrieval Strategy](docs/features/retrieval/three-stage-strategy.md)
- [Query UX](docs/QUERY_UX.md)
- [Managed Models](docs/MANAGED_MODELS.md)
- [CLI Reference](docs/CLI.md)
- [Python SDK](docs/SDK.md)
- [REST API](docs/API.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Unified Storage (SQLite + FTS5)](docs/features/platform/unified-storage.md)
- [Searchable Index & Background Enrichment](docs/features/platform/searchable-index-background-enrichment.md)
- [Ingestion Pipeline](docs/INGESTION.md)
- [Enrichment (Hierarchies, Entities)](docs/ENRICHMENT.md)
- [Epistemic Governance (Pyrrho)](docs/CONSTRAINTS.md)
- [Extension Points](docs/PLUGINS.md)
- [Feature Control](docs/FEATURE_CONTROL.md)
- [KRAG — Knowledge Routing Augmented Generation](docs/features/platform/krag.md)
- [Code Symbol Extraction](docs/features/ingestion/code-symbol-extraction.md)
- [Tabular Data Routing](docs/features/ingestion/tabular-data-routing.md)
- [Enterprise Gateway](docs/features/platform/enterprise-gateway.md)
- [Engines](docs/ENGINES.md)
- [Configuration Examples](docs/CONFIG_EXAMPLES.md)
- [Custom Engines](docs/CUSTOM_ENGINES.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
