# API Reference

Core data models and protocols for fitz-sage.

## Core Models

### EvidencePack

The retrieval-first response contract. `fitz retrieve` and
`fitz_sage.evidence()` return this shape.

For the product-level contract, Pyrrho metadata, and indexing-status examples,
see [Evidence Pack](EVIDENCE_PACK.md).

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

### RetrievalRun

The versioned execution record returned by `fitz_sage.trace()` and
`RetrievalEngine.trace(Query)`.

The public record groups stable contracts for query planning, strategy calls,
candidate stages, frozen compiled evidence, exact Pyrrho input and output,
selected `EvidencePack`, and environment fingerprints. It supports:

```python
run.to_dict(include_content=False)
run.to_json(include_content=False)
run.write(path, include_content=False)
RetrievalRun.from_dict(payload)
RetrievalRun.from_json(payload)
RetrievalRun.read(path)
run.explain()
```

Serialization is source-content-redacted by default. `include_content=True` is
required for `replay_pyrrho()`. See
[Retrieval Execution Records](RETRIEVAL_RUNS.md).

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

Query with KRAG hints:
```python
query = Query(
    text="Summarize the paper",
    metadata={"top_k": 8}
)
```

**Metadata Usage:**

The `metadata` field allows passing engine-specific parameters without breaking the paradigm-agnostic interface:
- **Fitz KRAG** reads `top_k` and `conversation_context` (the latter is used
  by configured query intelligence for conversational resolution).
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

**Runtime AnswerMode:**

Indicates how the runtime should treat the answer. Pyrrho v2's native model
verdict is exposed in governance metadata as `evidence_verdict`:
- `SUFFICIENT`: runtime mode for `SUFFICIENT` evidence
- `DISPUTED`: runtime mode for `DISPUTED` evidence
- `INSUFFICIENT`: runtime mode for `INSUFFICIENT` evidence

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
# FitzKragEngine.answer() requires a configured synthesizer.
answer = engine.answer(query)
print(answer.text)
```

**Implementation Notes:**

How the engine generates the answer is entirely up to the implementation:
- **Fitz KRAG**: Uses retrieval plus optional configured generation
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
    def point(
        self,
        source: Path,
        collection: str | None = None,
        *,
        start_worker: bool = True,
    ) -> Any: ...
    def wait_for_enrichment(self) -> None: ...
    def indexing_status(self) -> dict: ...
    def retrieve(self, query: Query) -> list: ...
    def evidence(self, query: Query) -> EvidencePack: ...
    def trace(self, query: Query) -> RetrievalRun: ...
```

**Usage:**
```python
from pathlib import Path
from fitz_sage import Query, create_engine

engine = create_engine("fitz_krag")
engine.load("docs")
engine.point(Path("./docs"), collection="docs")

# point() has already completed the searchable source index.
pack = engine.evidence(Query(text="Which documents are relevant?"))
run = engine.trace(Query(text="Why were these documents selected?"))
```
