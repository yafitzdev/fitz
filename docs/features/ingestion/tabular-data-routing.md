# Tabular Data Routing

## Problem

Tables in documents get chunked arbitrarily, breaking structure:

- **Q:** "How much does Alice earn?"
- **Standard RAG:** Returns fragments like "Alice" + "salary column" (separated chunks)
- **Expected:** Query the full table: `SELECT salary FROM employees WHERE name = 'Alice'`

Text-based retrieval (BM25, full-text, or embedding similarity) fails on entity-specific table queries because chunked text doesn't capture row-level data. Tables need **structured querying (SQL)**, not chunk retrieval.

## Solution: Table Registry + SQL Execution

Fitz stores tables in SQLite and registers them for guaranteed retrieval:

```
Q: "How much does Alice earn?"
     ↓
Table chunk retrieved via registry (guaranteed, not semantic similarity)
     ↓
LLM generates SQL: SELECT salary FROM employees WHERE name = 'Alice'
     ↓
SQL executed on stored table data
     ↓
Result: "Alice earns $85,000"
```

## How It Works

### At Ingestion

1. **Table detection** - Parser identifies tables in documents:
   - CSV files → full table
   - Markdown tables → embedded tables
   - PDF tables → extracted via Docling

2. **Table storage** - Tables stored in SQLite TableStore:
   - Each table gets a unique ID: `{source_file}:{table_index}`
   - Full table data stored in SQLite (not chunked)
   - Schema extracted: column names, types, sample rows

3. **Schema chunk indexing** - Schema chunks indexed for search:
   - Contains: table name, column names, sample rows (top 3)
   - Embedded and stored in vector DB
   - Tagged with `content_type: table_schema`

4. **Table registry** - Mapping of table IDs to source files:
   ```json
   {
     "employees.csv:0": "path/to/employees.csv",
     "report.md:1": "path/to/report.md"
   }
   ```

### At Query Time

1. **Schema chunk retrieval** - Semantic search retrieves relevant schema chunks

2. **Table loading** - Full table data loaded from SQLite TableStore

3. **SQL generation** - LLM generates SQL query:
   ```sql
   SELECT salary FROM employees WHERE name = 'Alice'
   ```

4. **SQL execution** - Query executed on in-memory SQLite table

5. **Result formatting** - LLM formats SQL results into natural language answer

## Key Design Decisions

1. **Always-on** - Tables are automatically detected and routed. No configuration needed.

2. **Guaranteed retrieval** - Table registry ensures tables are always retrieved when needed (not dependent on semantic similarity).

3. **Full table storage** - Tables stored intact in SQLite, not chunked and scattered.

4. **LLM-generated SQL** - Uses the same chat LLM to generate SQL (no separate query planner).

5. **In-memory execution** - SQLite tables loaded into memory for query execution (fast, no external DB).

## Configuration

No configuration required. Feature is baked into the ingestion and answering pipelines.

Internal parameters:
- `max_table_rows`: Max rows to index in schema chunk (default: 3 sample rows)
- `table_store_path`: `.fitz/tables/{collection}.db`

## Files

- **Table store:** `fitz_sage/ingestion/tables/table_store.py`
- **Table detection:** `fitz_sage/ingestion/parser/` (CSV, Markdown, Docling parsers)
- **SQL generation:** `fitz_sage/engines/fitz_krag/answering/table_query.py`
- **Table routing step:** `fitz_sage/engines/fitz_krag/retrieval/steps/table_query.py`

## Benefits

| Standard RAG | Tabular Data Routing |
|--------------|---------------------|
| Tables chunked arbitrarily | Tables stored intact |
| Row-level queries fail | Row-level queries via SQL |
| Semantic search on tables | Structured SQL queries |
| Headers separated from data | Full table structure preserved |

## Example

**Table:** `employees.csv`

| name  | salary | department |
|-------|--------|------------|
| Alice | 85000  | Engineering |
| Bob   | 75000  | Marketing |
| Carol | 90000  | Engineering |

### Query: "How much does Alice earn?"

**Standard RAG (no table routing):**
- Chunks:
  - Chunk 1: "name, salary, department"
  - Chunk 2: "Alice, 85000, Engineering"
  - Chunk 3: "Bob, 75000, Marketing"
- Problem: May not retrieve both header + Alice row together

**Tabular Data Routing:**
- Schema chunk retrieved: "employees.csv has columns: name, salary, department. Sample: Alice ($85,000), Bob ($75,000)"
- SQL generated: `SELECT salary FROM employees WHERE name = 'Alice'`
- SQL executed: Returns 85000
- Answer: "Alice earns $85,000"

### Query: "Who earns more than $80,000?"

**Standard RAG (no table routing):**
- May return partial list from limited chunks

**Tabular Data Routing:**
- SQL: `SELECT name FROM employees WHERE salary > 80000`
- Returns: Alice, Carol
- Answer: "Alice and Carol earn more than $80,000"

## Multi-Table Joins

Fitz supports JOIN queries across multiple tables:

**Tables:** `employees.csv` and `departments.csv`

**Query:** "Who works in the R&D department?"

**SQL generated:**
```sql
SELECT e.name
FROM employees e
JOIN departments d ON e.department = d.dept_id
WHERE d.dept_name = 'R&D'
```

## Dependencies

- `sqlite3` (built-in to Python)
- No external database required

## Performance Considerations

- **Ingestion:** Tables up to 10k rows handled efficiently
- **Query time:** <100ms for simple queries, <500ms for joins
- **Memory:** Tables loaded into memory (limit: ~50MB per table)

## Related Features

- **Keyword Vocabulary** - Exact matching helps find table names and column names
- **Multi-Hop Reasoning** - Can traverse table → references → other tables
- **Epistemic Honesty** - ABSTAIN if table doesn't contain requested data
