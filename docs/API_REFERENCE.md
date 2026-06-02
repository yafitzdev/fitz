# API Reference

Core data models and protocols for fitz-sage.

## Core Models

### EvidencePack

The retrieval-first response contract. `fitz query`, `fitz retrieve`, and
`fitz_sage.evidence()` return this shape.

**Definition:**
```python
@dataclass
class EvidencePack:
    query: str
    mode: AnswerMode | None
    items: list[EvidenceItem]
    reasons: list[str]
    timings: dict[str, float]
    indexing_status: dict[str, Any]
    metadata: dict[str, Any]
```

**Evidence items:**
```python
@dataclass
class EvidenceItem:
    rank: int
    source_id: str
    file_path: str
    address_kind: str
    address_location: str
    line_range: tuple[int, int] | None
    score: float | None
    excerpt: str
    content: str
    metadata: dict[str, Any]
```

Use `pack.to_dict()` or `pack.to_json()` when returning evidence from an API.

---

### Chunk

A generic unit of ingested content retained for compatibility with custom
engines and older plugin surfaces. The production KRAG engine uses typed
retrieval units (`Symbol`, `Section`, `TableSpec`) and exposes them through
`EvidenceItem`, not through public chunk objects.

**Definition:**
```python
@dataclass
class Chunk:
    id: str                   # Chunk ID
    doc_id: str              # Parent document ID
    content: str             # Chunk text content
    chunk_index: int         # Index within document
    metadata: dict[str, Any] # Optional metadata
```

**Usage:**
```python
chunk = Chunk(
    id="chunk_001",
    doc_id="doc_123",
    content="Quantum computing uses qubits...",
    chunk_index=0,
    metadata={"topic": "physics"}
)
```

---

### Query

Paradigm-agnostic query representation containing everything needed to ask a question.

**Definition:**
```python
@dataclass
class Query:
    text: str                          # The question being asked
    metadata: dict[str, Any]           # Engine-specific hints
```

**Examples:**

Simple query:
```python
query = Query(text="What is quantum computing?")
```

Query with engine hints:
```python
query = Query(
    text="Summarize the paper",
    metadata={"temperature": 0.3, "model": "claude-3-opus"}
)
```

**Metadata Usage:**

The `metadata` field allows passing engine-specific parameters without breaking the paradigm-agnostic interface:
- **Fitz KRAG** reads: `{"conversation_context": ...}` (for query rewriting / pronoun resolution)
- Custom engines can define their own metadata keys

Engines should ignore unknown metadata keys gracefully.

---

### Answer

Paradigm-agnostic answer representation encapsulating the response from a knowledge engine.

**Definition:**
```python
@dataclass
class Answer:
    text: str                      # The answer text
    provenance: list[Provenance]   # Sources used
    mode: AnswerMode | None        # Epistemic posture
    metadata: dict[str, Any]       # Engine metadata
```

**Examples:**

Simple answer:
```python
answer = Answer(text="Quantum computing uses qubits...")
```

Answer with sources:
```python
provenance = [
    Provenance(source_id="doc_1", excerpt="Qubits can be 0 and 1..."),
    Provenance(source_id="doc_2", excerpt="Quantum entanglement...")
]
answer = Answer(
    text="Quantum computing uses qubits which can exist in superposition...",
    provenance=provenance
)
```

Answer with epistemic mode:
```python
from fitz_sage.core.answer_mode import AnswerMode

answer = Answer(
    text="Sources disagree on this classification...",
    mode=AnswerMode.DISPUTED
)
```

Answer with engine metadata:
```python
answer = Answer(
    text="The answer is 42",
    metadata={
        "engine": "fitz_krag",
        "tokens_used": 1523,
        "confidence": 0.95
    }
)
```

**Provenance:**

The `provenance` field provides attribution and allows users to verify the
answer against source material. For retrieval-first workflows, prefer
`EvidencePack.items`, which include file path, location, score, excerpt, and
full content.

**Answer Mode:**

Indicates how certain the answer should be interpreted (3-class system):
- `TRUSTWORTHY`: Evidence clearly supports this answer
- `DISPUTED`: Sources disagree; answer presents multiple perspectives
- `ABSTAIN`: Insufficient evidence to answer definitively

If `None`, no epistemic assessment was performed.

**Metadata:**

Engine-specific metadata about answer generation can include:
- Performance metrics (tokens used, latency)
- Confidence scores
- Model information
- Reasoning traces
- Debug information

Consumers should be prepared for this to contain arbitrary data depending on the engine.

---

## Core Protocols

### KnowledgeEngine

Paradigm-agnostic protocol that all engines must implement.

**Philosophy:**
- Engines are black boxes that transform queries into answers
- Implementation details (retrieval, LLMs, reasoning) are engine-specific
- The platform only cares about: **Query in → Answer out**

**Protocol:**
```python
class KnowledgeEngine(Protocol):
    def answer(self, query: Query) -> Answer:
        """Execute a query against knowledge and return an answer."""
        ...
```

**Usage:**
```python
engine = FitzKragEngine(config)
query = Query(text="What is quantum computing?")
answer = engine.answer(query)
print(answer.text)
```

**Implementation Notes:**

How the engine generates the answer is entirely up to the implementation:
- **Fitz KRAG**: Uses retrieval + generation
- Custom engines might use completely different approaches

**Error Handling:**

Implementations should raise:
- `QueryError`: If the query is invalid or cannot be processed
- `KnowledgeError`: If knowledge retrieval/processing fails
- `EngineError`: For any other engine-specific errors

**Idempotency:**

Implementations should be idempotent when possible. The same query should produce consistent answers (though not necessarily identical, since LLMs may vary).

---

## Duck-Typed Protocols

### RetrievalEngine

Protocol implemented by engines that support persistent source registration and
retrieval-first evidence.

```python
class RetrievalEngine(KnowledgeEngine, Protocol):
    def load(self, collection: str) -> None: ...
    def point(self, source: Path, collection: str | None = None) -> Any: ...
    def wait_for_query_surface(self) -> None: ...
    def wait_for_indexing(self) -> None: ...
    def indexing_status(self) -> dict: ...
    def retrieve(self, query: Query) -> list: ...
    def evidence(self, query: Query) -> EvidencePack: ...
```

**Usage:**
```python
from pathlib import Path
from fitz_sage import Query, create_engine

engine = create_engine("fitz_krag")
engine.load("docs")
engine.point(Path("./docs"), collection="docs")
engine.wait_for_query_surface()

pack = engine.evidence(Query(text="Which documents are relevant?"))
```

---

### ChunkLike

Protocol for duck-typed chunk handling without requiring the concrete `Chunk` class.

**When to use:**
- You want to accept chunk-like objects from external sources
- You're building a plugin that needs flexibility
- You want to avoid coupling to the concrete Chunk class

**Properties:**
```python
class ChunkLike(Protocol):
    id: str
    doc_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] | None
```

**Note:** In most cases, using the concrete `Chunk` class is preferred. Only use `ChunkLike` when you need explicit duck-typing.
