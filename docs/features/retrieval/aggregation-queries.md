# Aggregation Queries

Fitz automatically detects and handles aggregation queries - questions asking for lists, counts, or enumerations.

## What It Does

When you ask questions like:
- "List all the test cases that failed"
- "How many errors occurred?"
- "What are the different types of reports?"
- "Enumerate all features of the system"

Fitz detects the aggregation intent and adjusts retrieval to get comprehensive coverage:

1. **Detects aggregation type**: LIST, COUNT, or UNIQUE
2. **Expands retrieval**: Fetches 3-4x more chunks than normal
3. **Augments query**: Adds instructions for exhaustive results
4. **Tags results**: Marks chunks with aggregation metadata for the answer generator

## Aggregation Types

| Type | Triggers | Fetch Multiplier |
|------|----------|------------------|
| **COUNT** | "how many", "count", "number of" | 4x |
| **UNIQUE** | "different", "distinct", "types of" | 3x |
| **LIST** | "list all", "enumerate", "what are the" | 3x |

## How It Works

```
Query: "List all the people mentioned in the documents"
         ↓
   [Aggregation Detector]
         ↓
   Detected: LIST intent
   Target: "people"
   Multiplier: 3x
         ↓
   [Augmented Query]
   "List all the people mentioned in the documents
    (include complete list of all people)"
         ↓
   [Expanded Retrieval]
   Fetch 75 chunks instead of 25
         ↓
   [Results Tagged]
   aggregation_type: LIST
   aggregation_target: people
```

## Example Queries

```
"List all test cases that failed"
→ LIST: fetches 3x chunks to find all failed tests

"How many errors occurred in the authentication module?"
→ COUNT: fetches 4x chunks for accurate counting

"What different types of reports are available?"
→ UNIQUE: fetches 3x chunks to find all report types

"Enumerate all the configuration options"
→ LIST: fetches 3x chunks for comprehensive listing
```

## Technical Details

- **Module**: `fitz_sage/retrieval/detection/modules/aggregation.py`
- **Integration**: invoked by the KRAG retrieval router (`fitz_sage/engines/fitz_krag/retrieval/router.py`)
- **No configuration**: Always active, no opt-out

### LLM-Based Detection

Detection uses a unified LLM classifier (one call for all detection types). The `AggregationModule` contributes its prompt fragment to the combined classification:

```python
# AggregationModule identifies:
# - COUNT: "how many", "count", "number of"
# - UNIQUE: "different types", "distinct", "kinds of"
# - LIST: "list all", "enumerate", "what are the"

# Returns DetectionResult with:
# - aggregation_type: COUNT | UNIQUE | LIST
# - target: what's being aggregated (e.g., "test cases")
# - fetch_multiplier: 3x-4x depending on type
```

### Query Handling

Each aggregation type triggers expanded retrieval:

- **COUNT**: 4x fetch multiplier for accurate counting
- **UNIQUE**: 3x fetch multiplier to find all distinct items
- **LIST**: 3x fetch multiplier for comprehensive listing

## Related Features

- [**Comparison Queries**](comparison-queries.md) - Multi-entity retrieval (another detection module)
- [**Temporal Queries**](temporal-queries.md) - Time-based retrieval (another detection module)
- [**Hierarchical RAG**](../ingestion/hierarchical-rag.md) - L2 summaries help with "list all themes" queries
