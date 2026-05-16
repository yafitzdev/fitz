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

## Solution: LLM-Powered Query Rewriting

Rewrite queries using conversation context before retrieval:

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

1. Query is sent to fast-tier LLM with conversation history
2. LLM performs transformations:
   - Resolves pronouns (it, they, this, that, their)
   - Fixes typos and removes filler words
   - Converts questions to document-matching form
   - Detects ambiguity
3. Rewritten query is used for retrieval
4. Original query is preserved for answer generation

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

1. **LLM-based** - Uses fast-tier chat model for intelligent rewriting.

2. **Single call** - All transformations in one LLM call for efficiency.

3. **Context-aware** - Maintains conversation history for pronoun resolution.

4. **Confidence scoring** - Each rewrite includes confidence (0.0-1.0).

5. **Preserves original** - Original query kept for fallback and answer generation.

6. **Graceful degradation** - On LLM failure, original query is used unchanged.

## Files

- **Rewriter module:** `fitz_sage/retrieval/rewriter/`
  - `rewriter.py` - QueryRewriter class
  - `types.py` - RewriteResult, ConversationContext
  - `prompts/rewrite.txt` - LLM prompt template
- **Integration:** `fitz_sage/engines/fitz_krag/retrieval/router.py`

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

- Uses the fast chat tier (any OpenAI-compatible model).
- One chat call per query — typically 100–300 ms on cloud, sub-second
  on a 7B local model.
- Skipped for very short queries (< 3 characters).
- On failure, falls back to the original query (no blocking).

## Dependencies

- A fast chat tier wired up via `chat_fast:`.
- No additional dependencies beyond the existing chat infrastructure.

## Related

- [Query Expansion](query-expansion.md) — rule-based synonym / acronym
  expansion that runs alongside LLM rewriting
- [Multi-Query RAG](multi-query-rag.md) — decomposes long queries into
  focused sub-queries
- [Sparse Search (FTS5 + bm25)](sparse-search.md) — the retrieval step
  the rewritten query feeds into
