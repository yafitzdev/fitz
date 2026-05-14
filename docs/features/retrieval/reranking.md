# Reranking (LLM-based)

## Problem

BM25 optimises for **recall** — find candidates that contain the
right tokens. But token overlap isn't the same as true relevance:

- Two sections can both mention the query terms while only one
  actually answers the question.
- The top-5 by `bm25()` aren't always the best 5 for the user's
  *intent*.
- BM25 is intent-blind: it can't tell a "how do I do X?" question
  from a "what is X?" question.

## Solution: LLM reranker

After FTS5 / BM25 returns a candidate set, ask the chat model to
score each `(query, candidate)` pair in a single JSON-returning
chat completion. Re-order by the model's score, keep the top-K.

```
Query: "What's the battery warranty?"
            │
            ▼
   FTS5 + bm25() — recall
   returns ~20 candidates
            │
            ▼
   LLMReranker — precision         ◀── enabled when `rerank:` is set
   one chat call scoring each pair
            │
            ▼
   Top 5 truly-relevant candidates
            │
            ▼
   Synthesizer + constraint cascade
```

This is **not a cross-encoder** model in the classical sense — it's
the same OpenAI-compatible chat protocol fitz-sage already speaks for
everything else. No separate reranker backend, no second SDK, no
embedding model.

## Why an LLM, not a cross-encoder?

The legacy reranker plugins (Cohere rerank-v3.5, BGE reranker, etc.)
were dropped in v0.12.0 along with the embedding stack. The reasoning:

1. **One protocol.** Adding a cross-encoder means adding a second
   network protocol, a second model to host, and a second auth path.
2. **Intent-aware.** A chat model sees the query and the candidate
   together with full instruction-following context. It can weight
   intent (`how do I` vs `what is`) the way a cross-encoder can't.
3. **Cheap.** One chat call to rank 10–20 candidates is fast — well
   under 2 s on a 7B local model, low-millisecond on cloud APIs.
4. **Already in the stack.** The synthesizer is a chat call; the
   query rewriter is a chat call; the detection step is a chat call.
   Reranking just rides the same path.

## How it works

```python
prompt = build_rerank_prompt(query, candidates)
# Sends a structured chat-completion request:
#   system: "You are scoring retrieval candidates for relevance ..."
#   user:   <query> + <numbered candidate snippets>
response_json = chat.chat([...])
# Returns: [{"id": "...", "score": 0.91, "why": "..."}]
```

The response is JSON. The reranker parses it, attaches `rerank_score`
to each candidate's metadata, and re-orders.

### Smart skip

If the candidate pool is small (default: `< 6`), the reranker step is
bypassed — there's nothing to rank.

### VIP preservation

Artifact rows (architecture narrative, dependency summary) carry a
sentinel `score = 1.0` and bypass the reranker. They're "always
include" by design.

## Key design decisions

1. **Provider-presence pattern.** Set `rerank: endpoint/llmreranker`
   and the step runs; omit (or set `null`) and it doesn't.
2. **Single chat call.** The whole candidate set goes in one prompt
   — saves per-candidate latency and lets the model see the candidates
   in context with each other.
3. **JSON contract.** The reranker speaks a documented JSON shape;
   parsing failures fall back to the original order.
4. **Metadata, not destructive.** `rerank_score` is added to each
   candidate's metadata; the original BM25 rank stays available.

## Configuration

### Enable

```yaml
rerank: endpoint/llmreranker
```

The reranker uses the configured chat tier (defaults to `chat_smart`).
Override:

```yaml
rerank: endpoint/llmreranker
rerank_tier: balanced       # use chat_balanced instead of chat_smart
```

### Disable

```yaml
# rerank: null    # or omit the key entirely
```

## Files

| Component                  | Path                                                              |
| -------------------------- | ----------------------------------------------------------------- |
| Pipeline step              | `fitz_sage/engines/fitz_krag/retrieval/reranker.py`               |
| Reranker chat provider     | `fitz_sage/llm/providers/llm_reranker.py`                         |
| Factory dispatch           | `fitz_sage/llm/config.py`                                         |

## Example

**Query:** "What's the warranty period for the battery?"

**After FTS5 + bm25 (top 5 by BM25):**

1. "Battery specifications: 75 kWh capacity ..."  (bm25 rank 1)
2. "Warranty terms vary by component ..."        (bm25 rank 2)
3. "The battery uses lithium-ion cells ..."      (bm25 rank 3)
4. "Battery warranty: 8 years or 100,000 miles." (bm25 rank 4)
5. "Charging the battery takes 45 minutes ..."   (bm25 rank 5)

**After LLM reranker:**

1. "Battery warranty: 8 years or 100,000 miles." (rerank 0.94)
2. "Warranty terms vary by component ..."        (rerank 0.78)
3. "Battery specifications: 75 kWh capacity ..." (rerank 0.61)
4. "The battery uses lithium-ion cells ..."      (rerank 0.45)
5. "Charging the battery takes 45 minutes ..."   (rerank 0.32)

The reranker correctly promotes the warranty-specific row over the
broader battery-spec candidates.

## Interaction with other features

| Feature                | Relationship                                                       |
| ---------------------- | ------------------------------------------------------------------ |
| Sparse search (FTS5)   | Runs *before* reranking; produces the candidate pool               |
| Query expansion        | Runs *before* reranking; all expanded results land in one pool     |
| KRAG routing           | Reranker is intent-aware via the query text it receives            |
| Multi-hop              | Reranker runs inside each hop independently                        |
| Constraint cascade     | Reranker output feeds the constraint cascade; reranker doesn't see constraints |

## Related

- [Sparse Search (FTS5 + bm25)](sparse-search.md) — the recall layer
- [Multi-Hop Reasoning](multi-hop-reasoning.md) — reranker runs inside each hop
- [Unified Storage](../platform/unified-storage.md) — SQLite + FTS5 layer
- [OpenAI-Compatible Endpoint](../platform/openai-compatible-endpoint.md) — the protocol everything speaks
