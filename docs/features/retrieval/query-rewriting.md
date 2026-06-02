<!-- docs/features/retrieval/query-rewriting.md -->
# Query Rewriting (Conversational Context Resolution)

## Problem

In conversational RAG, users frequently use pronouns and references that require context:

- "Tell me more about it" - What is "it"?
- "What about that company?" - Which company?
- "How does their authentication work?" - Whose?

Without conversation history, these queries retrieve nothing relevant.

Additional issues:
- Typos and filler words ("uhh, how do I like, fetch the config?")
- Complex phrasing that doesn't match document language
- Ambiguous queries with multiple possible meanings

## Solution: Optional LLM-Powered Query Rewriting

When `query_intelligence:` is configured, fitz-sage can rewrite queries using
conversation context before retrieval:

```
Conversation history:  User: "Tell me about TechCorp"
                       Assistant: "TechCorp is an EV company..."
                              ↓
Current query:         "What products do they make?"
                              ↓
Rewritten query:       "What products does TechCorp make?"
                              ↓
                       Search with resolved query
```

## How It Works

### Rewrite Types

| Type | Trigger | Example |
|------|---------|---------|
| **Conversational** | Pronouns with history | "Tell me about it" → "Tell me about TechCorp" |
| **Clarity** | Typos, filler words | "uhh how do I fetch config" → "how do I fetch config" |
| **Retrieval** | Question optimization | "What is X?" → "X definition overview" |
| **Combined** | Multiple issues | All of the above |

### At Query Time

1. Query is sent to the `query_intelligence` provider with conversation history
2. LLM performs transformations:
   - Resolves pronouns (it, they, this, that, their)
   - Fixes typos and removes filler words
   - Converts questions to document-matching form
   - Detects ambiguity
3. Rewritten query is used for retrieval
4. Original query is preserved for fallback and optional answer generation

### Ambiguity Detection

When a query has multiple possible meanings:

```
Query: "How do I handle errors?"
                ↓
Ambiguous: true
Disambiguated queries:
  - "How do I handle authentication errors?"
  - "How do I handle database connection errors?"
  - "How do I handle API request errors?"
```

All interpretations are searched and results are merged.

## Key Design Decisions

1. **Endpoint-backed enhancement** - Uses the `query_intelligence` chat provider for intelligent rewriting. With `query_intelligence: null`, the deterministic planner uses the original query.

2. **Batched** - Rewriting is one section of the single query-prep LLM call (alongside analysis, detection, keywords) — no call of its own.

3. **Context-aware** - Maintains conversation history for pronoun resolution.

4. **Confidence scoring** - Each rewrite includes confidence (0.0-1.0).

5. **Preserves original** - Original query kept for fallback and optional answer generation.

6. **Graceful degradation** - On LLM failure, original query is used unchanged.

## Files

- **Rewrite section:** `fitz_sage/engines/fitz_krag/query_batcher.py` (the `rewriting` section of the batched query-prep call)
- **Result parser:** `fitz_sage/retrieval/rewriter/rewriter.py` (`parse_rewrite_dict`)
- **Types:** `fitz_sage/retrieval/rewriter/types.py` (`RewriteResult`, `ConversationContext`)

## Benefits

| Without Rewriting | With Rewriting |
|-------------------|----------------|
| "What about it?" retrieves nothing | Resolves to actual topic |
| Typos cause misses | Typos are corrected |
| Pronouns break context | Pronouns resolved from history |
| Complex questions match poorly | Optimized for document matching |

## Example

**Conversation:**
```
User: "Tell me about the authentication system"
Assistant: "The system uses JWT tokens with 24-hour expiration..."
User: "How does it handle expired sessions?"
```

**Rewritten query:** "How does the authentication system handle expired sessions?"

**Result:** Documents about authentication session expiration are found, even though the user only said "it".

## Performance

- Runs inside the shared query-prep call (`query_intelligence`) — no chat
  call of its own.
- On failure or an empty rewrite, the original query is used unchanged.

## Dependencies

- A query-prep provider wired up via `query_intelligence:`.
- No additional dependencies beyond the existing chat infrastructure.

## Related

- [Query Expansion](query-expansion.md) — rule-based synonym / acronym
  expansion that runs alongside LLM rewriting
- [Multi-Query RAG](multi-query-rag.md) — decomposes long queries into
  focused sub-queries
- [Sparse Search (FTS5 + bm25)](sparse-search.md) — the retrieval step
  the rewritten query feeds into
