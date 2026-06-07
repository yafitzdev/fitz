<!-- README.md -->

<div align="center">

# fitz-sage

### Local-first governed retrieval for code, documents, and tables.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-sage.svg)](https://pypi.org/project/fitz-sage/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.15.0-green.svg)](CHANGELOG.md)
[![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)](https://github.com/yafitzdev/fitz-sage)

[EvidencePack](#evidencepack) • [Why `fitz-sage`?](#why-fitz-sage) • [Retrieval Intelligence](#retrieval-intelligence) • [Governance](#governance--know-what-you-dont-know) • [Documentation](#links) • [GitHub](https://github.com/yafitzdev/fitz-sage)

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
> `fitz query` runs locally by default. SQLite stores the index, ONNX models handle reranking and enrichment,
> and Pyrrho governs the returned evidence. An OpenAI-compatible endpoint is only needed when you choose
> generated prose with `fitz answer`.

```bash
pip install fitz-sage

# From a docs folder, --source is optional.
fitz query "What is our refund policy?" --source ./docs
```

The result is an `EvidencePack`: relevant source units, provenance, a governance verdict, and the signals needed to decide
what your application should do next.

---

### About

`fitz-sage` is a retrieval engine for local knowledge bases. It indexes code, documents, and tables into typed source units,
retrieves the units relevant to a question, reranks them, and returns a governed `EvidencePack` that downstream software can
inspect, display, store, or pass to a synthesizer.

⭐ The retrieval architecture is [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md). Code is parsed
as symbols, documents as sections, tables as SQLite-backed data, and fallback text as chunks. Queries are routed across those
typed surfaces with retrieval strategies that match the source structure.

⭐ Governance is enforced by [Pyrrho](https://huggingface.co/yafitzdev) in a local CPU forward pass. Pyrrho classifies the
query before retrieval, helps shape the first recall pass, and evaluates the selected evidence prefix as
`TRUSTWORTHY`, `DISPUTED`, or `ABSTAIN`.

Yan Fitzner — ([LinkedIn](https://www.linkedin.com/in/yan-fitzner/), [GitHub](https://github.com/yafitzdev), [HuggingFace](https://huggingface.co/yafitzdev)).

![fitz-sage honest_rag](https://raw.githubusercontent.com/yafitzdev/fitz-sage/main/docs/assets/honest_rag.jpg)

---

### Why `fitz-sage`?

**EvidencePack as the contract 🧾**
> Every query returns ranked source evidence, provenance, governance reasons, and retrieval metadata. Use it directly in
> APIs, CLIs, dashboards, agents, or pass it to an LLM for answer generation with calibrated governance.

**Asymmetric indexing 🗂️** → [KRAG (Knowledge Routing Augmented Generation)](docs/features/platform/krag.md)
> Source files become typed retrieval units: code symbols, document sections, tables, and fallback chunks. Each unit type
> keeps the structure needed to retrieve it well.

**Zero-wait querying 🐆** → [Progressive KRAG](docs/features/platform/progressive-krag-agentic-search.md)
> Ask immediately. `fitz-sage` builds the query-ready search surface first, returns governed evidence, and lets the
> background worker finish managed Qwen keyword/entity/hierarchy enrichment.

**Pyrrho-governed retrieval 🧭** → [Pyrrho docs](docs/CONSTRAINTS.md)
> Pyrrho profiles the query before retrieval, then judges the selected evidence after reranking. The query plan, 
> reasons, and missing-evidence signals travel with the `EvidencePack`, so callers know whether to answer, show conflict, 
> retrieve more, or ask for more source material.

**Queries that actually work 📊**
> Exact identifiers, temporal scopes, comparisons, aggregation requests, code lookups, table questions, and broad overview
> queries all flow through retrieval intelligence built into the engine.

**Tabular data that is actually searchable 📈** → [Unified Storage](docs/features/platform/unified-storage.md)
> CSVs and extracted tables live in SQLite with schema detection and SQL execution. Table queries use table structure,
> not arbitrary text fragments.

**Fully local execution possible 🏠**
> SQLite storage, ONNX reranking, managed Qwen enrichment, and Pyrrho governance all run locally. Optional synthesis can use
> any local or cloud OpenAI-compatible endpoint.

####

> [!TIP]
> Try fitz on itself:
>
> ```bash
> fitz query "How does the retrieval pipeline work?" --source ./fitz_sage
> ```

---

### What You Can Search

Traditional documents, source code, and tables have different structure. FitzKRAG preserves that structure during indexing
and retrieval.

<br>

| Retrieval Unit | Extracted From | How It Works |
|----------------|----------------|--------------|
| [**Symbols 🖌️**](docs/features/ingestion/code-symbol-extraction.md) | Code files | Tree-sitter parses functions, classes, and methods into addressable units with qualified names, references, and import graphs. |
| **Sections 📑** | Documents (PDF, markdown, text) | Headings and paragraphs become hierarchical sections with parent/child context and summaries. |
| [**Tables 📅**](docs/features/ingestion/tabular-data-routing.md) | CSV files or tables within documents | Native SQLite storage with schema detection and SQL execution from natural language. |
| **Images 🖼️** | Figures and diagrams within documents | VLM-powered figure extraction and visual understanding. *(Coming soon)* |
| **Chunks 🧩** | Any content as fallback | Fallback text retrieval when structured extraction does not apply. |

<br>

> [!NOTE]
> All retrieval units share the same retrieval intelligence: Pyrrho query profiling, temporal handling, comparisons,
> aggregation, keyword expansion, reranking, and governance cutoff.

---

### Retrieval Intelligence

[Retrieval Docs](docs/features/retrieval/README.md) • [Three-Stage Strategy](docs/features/retrieval/three-stage-strategy.md) • [Retrieval Pipeline](docs/RETRIEVAL_PIPELINE.md) • [Pre/Post Retrieval Signals](docs/features/retrieval/evidence-signals.md)

`fitz-sage` runs retrieval as a typed, governed pipeline:

<br>

| Stage | What it does | Main cost |
|-------|--------------|-----------|
| **1. Query profile + broad recall** | Pyrrho classifies the query contract, route, answerability shape, and modality. Fitz builds a retrieval profile, extracts query terms, adds semantic keywords, and searches typed units broadly. | SQLite FTS + deterministic planning + local Pyrrho query pass |
| **2. Rerank** | INT8 ONNX cross-encoder reorders broad candidates by query relevance. | Local ONNX reranker |
| **3. Pyrrho cutoff** | Pyrrho evaluates `query + top 1`, then `query + top 2`, and so on until the evidence prefix is trustworthy, disputed, or insufficient. | Local Pyrrho CPU classifier |

<br>

The quick path is keyword-first: exact query terms, Qwen semantic keywords, and BM25. Fully indexed collections add hierarchy,
entity graph links, corpus summaries, and richer context expansion.

[Built-in intelligence](docs/features/retrieval) handles the edge cases that break simple search:

<br>

| Feature | Query | What Fitz Uses |
|---------|-------|----------------|
| [**epistemic-honesty**](docs/features/governance/epistemic-honesty.md) | "What was our Q4 revenue?" | Pyrrho cutoff and abstain reasons |
| [**keyword-vocabulary**](docs/features/retrieval/keyword-vocabulary.md) | "Find TC_1000" | Exact identifier matching |
| [**sparse-search**](docs/features/retrieval/sparse-search.md) | "error code E_AUTH_401" | SQLite FTS5 + native `bm25()` |
| [**multi-hop**](docs/features/retrieval/multi-hop-reasoning.md) | "Who wrote the paper cited by the 2023 review?" | Iterative retrieval |
| [**hierarchical-rag**](docs/features/ingestion/hierarchical-rag.md) | "What are the design principles?" | Hierarchical summaries |
| [**multi-query**](docs/features/retrieval/multi-query-rag.md) | *[User pastes 500-char test report]* "What failed and why?" | Multi-query decomposition |
| [**comparison-queries**](docs/features/retrieval/comparison-queries.md) | "Compare React vs Vue performance" | Multi-entity coverage and comparison cutoff |
| [**entity-graph**](docs/features/retrieval/entity-graph.md) | "What else mentions AuthService?" | Entity-based linking across sources |
| [**temporal-queries**](docs/features/retrieval/temporal-queries.md) | "What changed between Q1 and Q2?" | Temporal scope detection |
| [**aggregation-queries**](docs/features/retrieval/aggregation-queries.md) | "List all the test cases that failed" | Exhaustive/list query handling |
| [**freshness-authority**](docs/features/retrieval/freshness-authority.md) | "What's the latest status on feature X?" | Recency and authority scoring |
| [**query-expansion**](docs/features/retrieval/query-expansion.md) | "How do I fetch the db config?" | Dictionary + managed-Qwen keyword expansion |
| [**query-rewriting**](docs/features/retrieval/query-rewriting.md) | "Tell me more about it" *(after discussing TechCorp)* | Conversational context resolution |
| [**reranking**](docs/features/retrieval/reranking.md) | "What's the battery warranty?" | ONNX cross-encoder reranker |

<br>

> [!IMPORTANT]
> Retrieval intelligence is baked in. Configuration declares providers; the engine decides which retrieval capabilities a
> query needs.

---

### EvidencePack

[Evidence Pack Contract](docs/EVIDENCE_PACK.md) • [Pre/Post Retrieval Signals](docs/features/retrieval/evidence-signals.md)

`EvidencePack` is the output contract of `fitz-sage`.

It gives you the relevant sources and the governance signals around them. You can show it directly, pass it to a model,
trigger a workflow from it, or store it as an audit artifact.

An `EvidencePack` has three parts:

| Part | Meaning | What you can do with it |
|------|---------|-------------------------|
| **Source evidence** | The documents, code symbols, table rows, sections, or chunks that matched the query. | Show citations, open source files, pass evidence into a model, or store provenance. |
| **Pre-retrieval profile** | Pyrrho's read of the user query before search starts. | Understand what kind of evidence Fitz looked for and why retrieval favored code, tables, sections, broad coverage, exact lookup, comparison coverage, etc. |
| **Post-retrieval verdict** | Pyrrho's judgment after seeing the retrieved evidence. | Decide whether to answer, abstain, retrieve more, ask for clarification, flag conflict, or request missing documents. |

#### Pre-retrieval signals

Before retrieval, Pyrrho classifies the query itself. These signals shape the first search.

| Signal | What it means | Why it matters |
|--------|---------------|----------------|
| `query_contract` | What evidence the query requires: exact lookup, comparison coverage, temporal grounding, exhaustive coverage, overview, or ordinary sufficiency. | Prevents a query that needs two sides, a timeframe, or a complete list from stopping on one plausible source. |
| `route` | The broad domain of the query: technical, legal, medical, financial, general, etc. | Tunes retrieval behavior to the kind of corpus and question. |
| `answerability_shape` | Whether the query looks like a direct answer, a synthesis, a set/list, or structured reasoning. | Controls how broad the search should be and how much evidence should be read. |
| `retrieval_modality` | The likely source surface: code, table, prose, config, logs, PDF layout, or mixed. | Pushes retrieval toward the source type most likely to contain the answer. |
| `top_k` / `top_read` | How much candidate evidence Fitz should collect and read. | Makes narrow lookups fast while giving broad or comparative questions enough coverage. |

#### Post-retrieval signals

After retrieval and reranking, Pyrrho evaluates the evidence prefix. These signals tell you whether the result is usable.

| Signal | What it means | What you can do with it |
|--------|---------------|-------------------------|
| `mode` | `TRUSTWORTHY`, `DISPUTED`, or `ABSTAIN`. | Gate generated answers, UI display, automation, or human review. |
| `reasons` | Plain-language explanation for the verdict. | Show users why Fitz trusted, disputed, or rejected the evidence. |
| `stop_reason` | Why retrieval stopped: enough evidence, exact lookup match, missing contract coverage, retry exhausted, etc. | Route the next step: answer, retry, broaden search, or ask for more source material. |
| `retrieval_action` | Pyrrho's next-action signal: answer now, retrieve more, broaden search, resolve conflict, ask clarification, or structured lookup. | Drive automatic second-pass retrieval or agent workflows. |
| `gap_type` | What is missing or unsafe: missing fact, timeframe, comparison side, authority, conflicting values, incomplete enumeration, ambiguous query, etc. | Tell the user exactly what evidence is needed. |
| `taxonomy` | The relationship between sources: direct answer, single authoritative source, consistent chain, conflict, partial overlap, wrong entity, and related cases. | Build better displays for agreement, conflict, authority, and source quality. |
| `scalars` | Continuous scores for sufficiency, query-evidence alignment, answer coverage, conflict density, retry value, false-trustworthy risk, and failure severity. | Monitor quality, tune workflows, trigger review, or decide when to retry. |

This is why `fitz-sage` is useful as infrastructure: the package returns source evidence plus enough judgment to decide the
next action.

---

### Governance — `Pyrrho`

[Feature docs](docs/CONSTRAINTS.md) • [Pyrrho on Hugging Face](https://huggingface.co/yafitzdev) • [fitz-gov benchmark](https://github.com/yafitzdev/fitz-gov)

Pyrrho is the local governance model behind `fitz-sage`. It runs in two places:

1. **Before retrieval** — classify `query_contract`, `route`, `answerability_shape`, and `retrieval_modality`.
2. **After retrieval** — evaluate whether the selected evidence prefix is trustworthy, disputed, or insufficient.

<br>

```
  Query
    │
    ▼
  Pyrrho query profile
    │  query_contract | route | answerability_shape | retrieval_modality
    ▼
  RetrievalProfile → broad recall → rerank
    │
    ▼
  Query + ranked evidence prefix
    │
    ▼
  Pyrrho evidence cutoff
    │
    ▼
  TRUSTWORTHY / DISPUTED / ABSTAIN → EvidencePack
```

<br>

| Signal | Purpose |
|--------|---------|
| `query_contract` | Evidence requirement: structured lookup, comparison coverage, temporal grounding, exhaustive coverage, representative overview, or evidence sufficiency. |
| `route` | Query domain for retrieval profile defaults. |
| `answerability_shape` | Direct answer, synthesis answer, set answer, or structured reasoning. |
| `retrieval_modality` | Expected source surface: text, table, code, config, log, PDF layout, or mixed. |
| `retrieval_action` | Evidence-stage guidance such as answer now, retrieve more, broaden search, resolve conflict, or structured lookup. |
| `gap_type` | Evidence-stage reason for missing or unsafe evidence. |
| `evidence_failure_severity` | Scalar severity signal for insufficient or unsafe evidence. |

The g4-alpha Pyrrho shape exposes the query and evidence heads Fitz needs for the full pre-retrieval and cutoff path. The
final g4 package should plug into the same model/config contract.

<br>

> [!NOTE]
> Governance is a source-evidence judgment. Pyrrho is trained to decide whether retrieved evidence supports trust,
> conflict, or abstention, and Fitz records that judgment in the returned metadata.

<strong>The system fails safe 🛡️</strong>
> Threshold calibration is tuned around avoiding false trustworthy decisions. When evidence is incomplete or conflicting,
> the returned mode and reasons make that explicit.

<strong>No LLM on the governance path ⏱️</strong>
> Pyrrho is a local encoder forward pass. Governance does not require an external chat model.

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
>`fitz-sage` creates a local retrieval config on first run:
>1. **SQLite storage** for collections.
>2. **Managed ONNX models** for reranking and enrichment.
>3. **Pyrrho governance** for query profile and evidence cutoff.
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
>pack = fitz_sage.evidence("Where is classify_query implemented?", source="./fitz_sage")
>
>print(pack.mode)
>for item in pack.items:
>    print(item.file_path, item.address_location)
>```
>
>The SDK provides:
>- Module-level `evidence()` matching `fitz query`
>- Module-level `query()` for generated prose from evidence
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
>fitz query "Your question here" --source ./docs
>```
>
>Reranking, governance, and required enrichment run locally. No data leaves your machine for `fitz query`.
>
>Optional synthesis can use [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai),
>[Ollama](https://ollama.ai) in `/v1/` mode, [TabbyAPI](https://github.com/theroyallab/tabbyAPI), OpenAI, Together,
>Groq, Fireworks, OpenRouter, or any endpoint that speaks the OpenAI HTTP protocol.

</details>

---

<details>

<summary><strong>📦 Real-World Usage</strong></summary>

<br>

`fitz-sage` is a retrieval foundation. It manages indexing, search, reranking, governance, and provenance so products can
build on source evidence.

<br>

<strong>Chatbot Backend 🤖</strong>

> Connect fitz to Slack, Discord, Teams, or your own UI. The bot can show source-backed evidence, ask for more documents
> when Pyrrho abstains, or call `fitz answer` for generated prose.
>
> *Example:* A support bot retrieves policy sections, shows links to the relevant docs, and only synthesizes when the
> evidence pack is trustworthy.

<br>

<strong>Internal Knowledge Base 📖</strong>

> Point fitz at your wiki, policies, runbooks, and repos. Employees ask natural-language questions and get source units
> with provenance.
>
> *Example:* New hires ask "How do I request PTO?" and receive the exact policy section plus the governance verdict.

<br>

<strong>Continuous Intelligence & Alerting (Watchdog) 🐶</strong>

> Run scheduled queries over changing folders, logs, reports, or exports. Trigger alerts when the evidence pack contains
> trusted sources, disputes, or missing-coverage signals.
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
│  CLI: query | retrieve | answer | collections | serve           │
│  SDK: fitz_sage.evidence(source=...)                            │
│  API: /query | /chat | /collections | /health                   │
├─────────────────────────────────────────────────────────────────┤
│  Engine                                                         │
│  FitzKRAG: typed retrieval over code, docs, tables, chunks       │
├─────────────────────────────────────────────────────────────────┤
│  Evidence Contract                                              │
│  EvidencePack: items | mode | reasons | timings | metadata      │
├─────────────────────────────────────────────────────────────────┤
│  Pyrrho                                                         │
│  query profile | retrieval action | evidence cutoff             │
├─────────────────────────────────────────────────────────────────┤
│  Local CPU Models                                               │
│  ONNX reranker | managed Qwen enrichment | Pyrrho governance    │
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
fitz query "question" --source ./docs     # Return governed evidence
fitz query "question"                     # Use current folder or existing collection
fitz retrieve "question" --format json    # Evidence with script-friendly controls
fitz answer "question" --synthesizer ...  # Generated prose from evidence
fitz collections                          # List and delete knowledge collections
fitz serve                                # Start REST API server
```

Config: `~/.fitz/config/fitz_krag.yaml` — auto-created on first run. Edit it for optional synthesis, query intelligence,
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
pack = fitz_sage.evidence("Where is classify_query implemented?", source="./fitz_sage")

print(pack.mode)  # TRUSTWORTHY, DISPUTED, or ABSTAIN

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

fitz serve                    # localhost:8000
fitz serve -p 3000            # custom port
fitz serve --host 0.0.0.0     # all interfaces
```

**Interactive docs:** Visit `http://localhost:8000/docs` for Swagger UI.

<br>

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Return a governed evidence response |
| POST | `/chat` | Return generated prose from retrieved evidence |
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
> Your Python Scripts directory is not on PATH. Use `python -m fitz_sage.cli.cli`, or add the Scripts directory to PATH.

**PDF/DOCX files are being skipped**
> Document parsing requires the optional document parser dependencies. Install them with:
> `pip install fitz-sage[docs]`

**"Connection refused at localhost:8080" error**
> This applies to optional endpoint-backed synthesis or query intelligence. `fitz query "..."` returns evidence without an
> endpoint server. For generated prose:
> `fitz answer "..." --synthesizer openai/gpt-4o`.

**"Model not found" error**
> The model name in your config does not match what your server has loaded. Check `/v1/models` on your server:
> `curl http://localhost:8080/v1/models`. Then update `synthesizer` in `~/.fitz/config/fitz_krag.yaml`.

**First query is slow**
> First run initializes storage, downloads managed local models if needed, and builds the query-ready index. Later queries
> reuse the collection.

**How do I change my LLM endpoint or model?**
> Edit `~/.fitz/config/fitz_krag.yaml`:
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

**Documentation:**
- [Docs Index](docs/README.md)
- [Evidence Pack Contract](docs/EVIDENCE_PACK.md)
- [Pre/Post Retrieval Signals](docs/features/retrieval/evidence-signals.md)
- [Three-Stage Retrieval Strategy](docs/features/retrieval/three-stage-strategy.md)
- [Query UX](docs/QUERY_UX.md)
- [Managed Models](docs/MANAGED_MODELS.md)
- [CLI Reference](docs/CLI.md)
- [Python SDK](docs/SDK.md)
- [REST API](docs/API.md)
- [Configuration Guide](docs/CONFIG.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Unified Storage (SQLite + FTS5)](docs/features/platform/unified-storage.md)
- [Progressive KRAG & Agentic Search](docs/features/platform/progressive-krag-agentic-search.md)
- [Ingestion Pipeline](docs/INGESTION.md)
- [Enrichment (Hierarchies, Entities)](docs/ENRICHMENT.md)
- [Epistemic Governance (Pyrrho)](docs/CONSTRAINTS.md)
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
