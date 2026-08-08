<!-- docs/EVIDENCE_PACK.md -->
# Evidence Pack

`EvidencePack` is the retrieval-first response contract. It is returned by:

- `fitz retrieve`
- `fitz_sage.evidence()`

It is intentionally not an answer. It is ranked, governed evidence that another
application can inspect, display, store as audit data, or pass into optional
synthesis.

For the product use of retrieval and governance metadata, see
[Evidence Signals](features/retrieval/evidence-signals.md).

## Shape

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

Use `pack.to_dict()` or `pack.to_json()` for API responses.

## Modes

| Runtime mode | Meaning |
|---|---|
| `sufficient` | Runtime API mode for a Pyrrho `SUFFICIENT` verdict. |
| `disputed` | Runtime API mode for a Pyrrho `DISPUTED` verdict. |
| `insufficient` | Runtime API mode for a Pyrrho `INSUFFICIENT` verdict. |
| `null` | Governance did not run. This is not the default product path. |

## Metadata

The most important metadata blocks are:

| Block | Meaning |
|---|---|
| `query_profile` | The effective retrieval profile used before recall. |
| `retrieval_trace` | Candidate generation, reranking, final reads, and evidence-closure passes. |
| `evidence_compiler` | Mechanical evidence roles, anchors, and ordering. |
| `evidence_delivery` | Progressive prefix sizes and exact Pyrrho trajectory. |
| `pyrrho` | Pyrrho's exact serialized governance decision. |

### Query Profile

`metadata.query_profile` records how Fitz searched before governance ran. It
contains query-shape metadata, managed Qwen query keywords, strategy weights,
fetch limits, and intent flags.
The `has_*_intent` fields describe Fitz's deterministic reading of the user's
query. Exact Pyrrho PRE evidence obligations remain separate in
`query_profile.pyrrho_pre` and `retrieval_intents`.

```json
{
  "metadata": {
    "query_profile": {
      "profile": {
        "domain": "technical",
        "specificity": "moderate",
        "answer_type": "comparative",
        "top_k": 50,
        "top_read": 50,
        "strategy_weights": {
          "code": 0.25,
          "section": 0.25,
          "table": 0.55
        },
        "keywords": ["incident", "eu", "release"],
        "comparison_entities": ["EU token rotation", "policy interval"],
        "has_comparison_intent": true,
        "has_temporal_intent": true
      }
    }
  }
}
```

### Evidence Delivery And Pyrrho

Fitz-Sage starts with the first three compiled evidence items and adds two only
after Pyrrho returns exact `INSUFFICIENT`. Exact `SUFFICIENT` or `DISPUTED`
stops immediately. `metadata.evidence_delivery` records the mechanical schedule
and every exact Pyrrho output.
`metadata.pyrrho` is the dictionary returned by Pyrrho without reinterpretation.

```json
{
  "metadata": {
    "evidence_delivery": {
      "available": 7,
      "selected": 5,
      "limit": 7,
      "initial_prefix_size": 3,
      "prefix_increment": 2,
      "evaluated_prefixes": [3, 5]
    },
    "pyrrho": {
      "schema_version": 1,
      "verdict": "SUFFICIENT",
      "reason": "Pyrrho: evidence is sufficient for a confident answer (P=0.93).",
      "probabilities": {
        "INSUFFICIENT": 0.03,
        "DISPUTED": 0.04,
        "SUFFICIENT": 0.93
      },
      "heads": {
        "evidence_verdict": {
          "final_label": "SUFFICIENT",
          "confidence": 0.93
        },
        "failure_mode": {
          "final_label": "none",
          "confidence": 0.91
        },
        "retrieval_intents": {
          "final_labels": ["needs_comparison_or_set"],
          "confidence": 0.82
        },
        "evidence_kinds": {
          "final_labels": ["needs_text", "needs_table_or_record"],
          "confidence": 0.88
        }
      },
      "input": {
        "tokens": 834,
        "truncated": false,
        "max_tokens": 2048
      }
    }
  }
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `evidence_delivery.available` | Compiled evidence items available before the delivery budget. |
| `evidence_delivery.selected` | Items in the stopping prefix returned in the pack. |
| `evidence_delivery.limit` | Maximum prefix from requested `top_k`, or configured `top_read`. |
| `evidence_delivery.initial_prefix_size` | First evaluation target: three items, or all available when fewer. |
| `evidence_delivery.prefix_increment` | Additional ranked items after each `INSUFFICIENT`: two. |
| `evidence_delivery.evaluated_prefixes` | Prefix sizes Pyrrho actually evaluated. |
| `evidence_delivery.trajectory` | Each prefix size and Pyrrho's exact serialized decision for it. |
| `pyrrho.verdict` | Authoritative native verdict. |
| `pyrrho.probabilities` | Native verdict probabilities. |
| `pyrrho.heads` | Native v2 verdict, failure, retrieval-intent, and evidence-kind heads. |
| `pyrrho.input` | Token count, truncation status, and model token limit. |

Pyrrho owns thresholds and contradictory-head consistency. Fitz-Sage neither
adds confidence thresholds nor overrides the verdict for any query shape; it
only continues when the exact verdict is `INSUFFICIENT`.

### Retrieval Trace

`metadata.retrieval_trace` is included so benchmark and analysis tools can
inspect how a pack was produced:

```json
{
  "metadata": {
    "retrieval_trace": {
      "query": "Where is expired session refresh implemented?",
      "profile": {},
      "router": {
        "tagged_queries": [],
        "strategy_calls": [],
        "raw_candidates": [],
        "deduped": [],
        "ranked": [],
        "final": []
      },
      "reranker": {
        "used": true,
        "input": [],
        "output": []
      },
      "final_addresses": [],
      "read_results": []
    }
  }
}
```

The trace is not a separate debug API. It is part of the retrieval-first
contract because benchmark reports need candidate frontiers, strategy scores,
reranker order, and evidence-closure behavior alongside the selected evidence.

### Evidence Compiler

`metadata.evidence_compiler` records mechanical evidence constraints before
progressive delivery: exact identifiers, soft keyword anchors, how many evidence
items entered and left compilation, and selected evidence roles.

```json
{
  "metadata": {
    "evidence_compiler": {
      "contract": {
        "identifiers": ["INC-101"],
        "keyword_anchors": ["latest", "status"],
        "required_modalities": ["table"],
        "temporal_policy": "temporal"
      },
      "input_count": 4,
      "output_count": 2,
      "filtered_all": false,
      "selected": []
    }
  }
}
```

## Indexing Status

`indexing_status` describes whether the collection can answer retrieval queries
and whether optional enrichment is complete.

```json
{
  "indexing_status": {
    "total": 65,
    "indexed": 64,
    "pending": 0,
    "failed": 1,
    "complete": false,
    "query_ready": true,
    "by_index_state": {
      "indexed": 64,
      "failed": 1
    },
    "enrichment": {
      "total": 64,
      "completed": 42,
      "pending": 22,
      "failed": 0,
      "finalization": "pending",
      "complete": false
    }
  }
}
```

| Field | Meaning |
|---|---|
| `total` | Supported files, including source-index failures. |
| `indexed` | Files stored in the searchable source index. |
| `pending` | Files not yet settled by source indexing. |
| `failed` | Supported files that failed source indexing. |
| `complete` | Every supported file indexed successfully. |
| `query_ready` | No supported file remains pending. |
| `by_index_state` | Source-index state counts. |
| `enrichment` | Independent entity/hierarchy progress and failures. |

## Item Metadata

Evidence items are typed retrieval units. `address_kind` identifies the unit:

| Kind | Typical source |
|---|---|
| `section` | Markdown, PDF, DOCX, PPTX, HTML, text, or parsed prose section. |
| `symbol` | Code function, class, method, constant, or module-level symbol. |
| `table` | Native configured delimited table, `.csv`/`.tsv` by default. |
| `file` | Code file selected or added through structural expansion. |

`excerpt` is display text. `content` is the fuller text passed into Pyrrho and
optional synthesis.
