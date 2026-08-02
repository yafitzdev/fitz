# Code Symbol Extraction

KRAG indexes code as addressable symbols instead of fixed-size text windows.
The source file remains the authority; symbol rows point to line ranges that are
read from the stored raw source on demand.

## Supported Languages

| Language | Primary parser | Typical units |
|---|---|---|
| Python | standard-library `ast` | modules, functions, classes, methods, constants |
| TypeScript/JavaScript | tree-sitter | functions, classes, methods, interfaces, types |
| Go | tree-sitter | functions, methods, structs, interfaces, constants/variables |
| Java | tree-sitter | classes, interfaces, enums, records, methods, fields |

When the primary parser cannot run, language strategies use their documented
regex fallback. That keeps the file indexable with reduced structural detail;
it does not claim full parser equivalence.

## Symbol Contract

Each `SymbolEntry` records:

```python
@dataclass
class SymbolEntry:
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str = ""
    imports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
```

Full source is stored once in `krag_raw_files`, not duplicated into every symbol
row.

## Search Surface

`krag_symbol_index` stores names, qualified names, ranges, signatures,
references, and optional background entities. Its external-content FTS5 table
indexes one derived `index_text` containing:

- the original name and qualified name;
- camel-case, dotted, underscored, and hyphen-separated name tokens;
- the signature;
- the source-authored docstring or doc comment.

There are no generated symbol summaries, keyword arrays, or alias mappings in
the source-index path. Identifier tokenization improves lexical search but does
not assert that differently formatted domain IDs are equivalent.

Two recall legs feed the common pool:

1. FTS5 BM25 over `index_text`;
2. case-insensitive substring lookup over symbol and qualified names.

The ONNX reranker then scores the bounded candidate prefix.

## Import And Reference Context

Language strategies also emit file-level import edges. After all changed files
are indexed, `resolve_imports()` connects targets that exist in the collection.
The expander can use resolved imports, same-file references, class context, and
optional entity links to add bounded neighboring evidence.

This is retrieval context expansion, not a complete static-analysis API.
Dynamic imports, reflection, generated code, and unresolved external modules
remain outside the graph.

## Background Entity Links

Optional enrichment can extract entities from symbols and populate the entity
graph. A failure leaves names, signatures, docstrings, imports, and raw source
fully searchable.

## Implementation

- `fitz_sage/engines/fitz_krag/ingestion/strategies/base.py`
- `fitz_sage/engines/fitz_krag/ingestion/strategies/python_code.py`
- `fitz_sage/engines/fitz_krag/ingestion/strategies/typescript.py`
- `fitz_sage/engines/fitz_krag/ingestion/strategies/go.py`
- `fitz_sage/engines/fitz_krag/ingestion/strategies/java.py`
- `fitz_sage/engines/fitz_krag/ingestion/symbol_store.py`
- `fitz_sage/engines/fitz_krag/ingestion/import_graph_store.py`

## Related

- [KRAG](../platform/krag.md)
- [Sparse Search](../retrieval/sparse-search.md)
- [Entity Graph](../retrieval/entity-graph.md)
