# Comparison Queries

## Problem

Questions comparing two entities fail when only one is retrieved:

- **Q:** "Compare React vs Vue performance"
- **Standard RAG:** Returns only React docs (misses Vue entirely)
- **Result:** Incomplete comparison, one-sided answer

Semantic search with a single query often favors one entity over the other. Balanced comparisons need **guaranteed retrieval of both entities**.

## Solution: Multi-Entity Retrieval

Fitz detects comparison intent and ensures both entities are retrieved:

```
Q: "Compare React vs Vue performance"
     ↓
Detected: COMPARISON intent, entities: [React, Vue]
     ↓
Separate searches:
  → "React performance"
  → "Vue performance"
     ↓
Both entity sets merged → complete comparison data
```

## How It Works

### LLM-Based Comparison Detection

Detection rides the single batched `QueryBatcher` LLM call (one call for analysis + all detection types + rewriting + keywords). The `ComparisonModule` identifies comparison queries and extracts entities:

```python
# ComparisonModule identifies:
# - "X vs Y", "compare X and Y", "difference between X and Y"
# - Returns: entities being compared, comparison context

# Example DetectionResult:
# - detected: True
# - comparison_entities: ["React", "Vue"]
# - comparison_queries: ["React performance", "Vue performance"]
```

### Entity Extraction

1. **Parse query** - Extract entity names using patterns
2. **Validate entities** - Ensure both entities are substantive (not stop words)
3. **Generate sub-queries** - Create focused queries for each entity:
   ```
   Original: "Compare React vs Vue performance"
   Sub-queries:
     → "React performance"
     → "Vue performance"
   ```

### Multi-Entity Retrieval

1. **Parallel search** - Execute separate BM25/FTS5 searches for each entity
2. **Merge results** - Combine chunks from both searches
3. **Deduplicate** - Remove overlapping chunks (e.g., docs mentioning both)
4. **Answer generation** - LLM receives balanced context from both entities

## Key Design Decisions

1. **Always-on** - Comparison detection is baked into query processing. No configuration needed.

2. **Parallel searches** - Entity searches run concurrently (no sequential bottleneck).

3. **Graceful fallback** - If detection fails, falls back to standard search.

4. **Entity-agnostic** - Works for any entities (frameworks, products, concepts, people, etc.).

5. **Preserves original query** - Original query also searched (in addition to entity sub-queries) to catch cross-entity comparisons.

## Configuration

No configuration required. Feature is baked into the retrieval pipeline.

Internal parameters:
- `min_entity_length`: Minimum characters for valid entity (default: 2)
- `max_entities`: Maximum entities to extract (default: 2)

## Files

- **Detection module:** `fitz_sage/retrieval/detection/modules/comparison.py`
- **Query intelligence:** `fitz_sage/engines/fitz_krag/query_batcher.py` (`QueryBatcher`)
- **Integration:** `fitz_sage/engines/fitz_krag/retrieval/router.py`

Detection is LLM-based. The `ComparisonModule` contributes its prompt fragment to the batched `QueryBatcher` call, then extracts entities and generates entity-specific sub-queries.

## Benefits

| Standard RAG | Comparison Queries |
|--------------|-------------------|
| May only retrieve one entity | Guaranteed both entities retrieved |
| One-sided comparisons | Balanced, complete comparisons |
| Depends on query phrasing | Robust to phrasing variations |

## Example

**Query:** "Compare React vs Vue performance"

**Standard RAG (no comparison detection):**
- Single search: "Compare React vs Vue performance"
- Top 5 results:
  1. "React's virtual DOM provides excellent performance..."
  2. "React reconciliation is highly optimized..."
  3. "React performance benchmarks show..."
  4. "Many frameworks like Vue and Angular..."
  5. "React is widely used for SPAs..."
- Problem: Only React-focused docs retrieved

**Comparison Queries:**
- Detected: COMPARISON intent, entities: [React, Vue]
- Search 1: "React performance" → 5 React docs
- Search 2: "Vue performance" → 5 Vue docs
- Search 3: "Compare React vs Vue performance" → 5 cross-comparison docs
- Merged: 10-15 unique chunks (balanced React + Vue content)
- Answer: "React uses a virtual DOM with reconciliation, achieving X ms render times. Vue uses a reactivity system with dependency tracking, achieving Y ms render times. Both are performant, but React excels at..."

## Detected Comparison Patterns

| Pattern | Example |
|---------|---------|
| **X vs Y** | "Python vs Java", "AWS vs Azure" |
| **X versus Y** | "REST versus GraphQL" |
| **compare X and Y** | "compare microservices and monoliths" |
| **compare X to Y** | "compare Redis to Memcached" |
| **difference between X and Y** | "difference between async and sync" |
| **X or Y** | "Should I use Postgres or MySQL?" |

## Edge Cases

### More than 2 entities

**Q:** "Compare React, Vue, and Angular performance"

- Currently: Extracts first 2 entities (React, Vue)
- Future: Support N-way comparisons

### Nested comparisons

**Q:** "Compare React Hooks vs Vue Composition API for state management"

- Extracts: [React Hooks, Vue Composition API]
- Context: "state management" included in both sub-queries

### Implicit comparisons

**Q:** "Is React faster than Vue?"

- Pattern: "X faster than Y"
- Extracts: [React, Vue]
- Works correctly

## Dependencies

- Requires a chat LLM client for detection (the batched `QueryBatcher` call)
- Part of the combined LLM classification call (no additional latency)

## Performance Considerations

- **Latency:** +0-500ms (parallel searches are concurrent, minimal overhead)
- **Retrieval count:** 2-3x normal (entity1 + entity2 + original query)
- **LLM context:** Larger context (more chunks), slightly higher cost

## Related Features

- **Multi-Query** - Long queries decomposed; comparison is a special case of decomposition
- **Query Expansion** - Synonyms/acronyms expanded; comparison adds entity-specific sub-queries
- **Temporal Queries** - Period filtering; comparison is entity filtering
