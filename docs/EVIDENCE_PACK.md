<!-- docs/EVIDENCE_PACK.md -->
# Evidence Pack

`EvidencePack` is the retrieval-first response contract. It is returned by:

- `fitz query`
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
| `trustworthy` | Runtime API mode for a Pyrrho `SUFFICIENT` verdict. |
| `disputed` | Runtime API mode for a Pyrrho `DISPUTED` verdict. |
| `abstain` | Runtime API mode for a Pyrrho `INSUFFICIENT` verdict. |
| `null` | Governance did not run. This is not the default product path. |

## Metadata

The most important metadata blocks are:

| Block | Meaning |
|---|---|
| `query_profile` | The effective retrieval profile used before recall. |
| `retrieval_trace` | Candidate generation, reranking, final reads, and retries. |
| `evidence_compiler` | Mechanical evidence roles, anchors, and source-count constraints. |
| `governance_cutoff` | Pyrrho v2 prefix evaluation and final governance decision. |

### Query Profile

`metadata.query_profile` records how Fitz searched before governance ran. It
contains query-shape metadata, managed Qwen query keywords, strategy weights,
fetch limits, and intent flags.

```json
{
  "metadata": {
    "query_profile": {
      "signals": {},
      "profile": {
        "domain": "technical",
        "specificity": "moderate",
        "answer_type": "comparative",
        "top_k": 50,
        "top_read": 50,
        "strategy_weights": {
          "code": 0.25,
          "section": 0.25,
          "table": 0.55,
          "chunk": 0.35
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

### Governance Cutoff

`metadata.governance_cutoff` records how Pyrrho evaluated ranked evidence
prefixes. The runtime `mode` stays in the `trustworthy` / `disputed` /
`abstain` vocabulary, while Pyrrho v2 heads expose the model's native evidence
metadata.

```json
{
  "metadata": {
    "governance_cutoff": {
      "evaluated": 3,
      "selected": 3,
      "max": 10,
      "mode": "trustworthy",
      "stop_reason": "trustworthy_min_evidence_met",
      "policy": {
        "query_shape": "comparison",
        "min_trustworthy_docs": 2,
        "min_disputed_docs": 2,
        "disputed_patience_docs": 2
      },
      "pyrrho": {
        "mode": "trustworthy",
        "probabilities": {
          "abstain": 0.03,
          "disputed": 0.04,
          "trustworthy": 0.93
        },
        "reason": "Pyrrho: sources support a confident answer (P=0.93).",
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
      }
    }
  }
}
```

Field meanings:

| Field | Meaning |
|---|---|
| `evaluated` | How many ranked evidence prefixes Pyrrho evaluated. |
| `selected` | How many evidence items were returned after cutoff. |
| `max` | Maximum cutoff window for this query, capped at 10 by default. |
| `mode` | Final runtime governance mode for the selected prefix. |
| `stop_reason` | Why the cutoff loop stopped. |
| `policy.query_shape` | Narrow, broad, comparison, or aggregation. |
| `policy.min_trustworthy_docs` | Minimum prefix size before the runtime `trustworthy` mode can stop. |
| `policy.min_disputed_docs` | Minimum prefix size before comparison disputes can stop. |
| `policy.disputed_patience_docs` | Additional patience for narrow disputes. |
| `pyrrho.probabilities` | Runtime probabilities for `abstain`, `disputed`, and `trustworthy` modes. |
| `pyrrho.evidence_verdict` | Native v2 verdict head. |
| `pyrrho.failure_mode` | Native v2 failure-mode head. |
| `pyrrho.retrieval_intents` | Native v2 retrieval-intent head. |
| `pyrrho.evidence_kinds` | Native v2 evidence-kind head. |

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
      "read_results": [],
      "retries": []
    }
  }
}
```

The trace is not a separate debug API. It is part of the retrieval-first
contract because benchmark reports need candidate frontiers, strategy scores,
reranker order, and retry behavior alongside the selected evidence.

### Evidence Compiler

`metadata.evidence_compiler` records mechanical evidence constraints before
Pyrrho cutoff: literal anchors, required source count, how many evidence items
entered and left compilation, and selected evidence roles.

```json
{
  "metadata": {
    "evidence_compiler": {
      "contract": {
        "identifiers": ["INC-101"],
        "phrase_anchors": ["Project Orion"],
        "source_anchors": [],
        "keyword_anchors": ["latest", "status"],
        "metric_terms": [],
        "required_modalities": ["table"],
        "temporal_policy": "temporal"
      },
      "input_count": 4,
      "output_count": 2,
      "min_sources": 2,
      "filtered_all": false,
      "selected": []
    }
  }
}
```

## Indexing Status

`indexing_status` describes whether the collection can answer retrieval queries
and whether required deep enrichment is complete.

```json
{
  "indexing_status": {
    "total": 65,
    "indexed": 64,
    "pending": 1,
    "complete": false,
    "query_ready": false,
    "deep_pending": 22,
    "fully_enriched": false,
    "by_state": {
      "query_ready": 43,
      "enriched": 21,
      "registered": 1
    }
  }
}
```

| Field | Meaning |
|---|---|
| `total` | Files tracked by the collection manifest. |
| `indexed` | Files in a query-ready state. |
| `pending` | Files not query-ready yet. |
| `complete` | `pending == 0`; the query-ready index is complete. |
| `query_ready` | Same readiness signal as `complete`. |
| `deep_pending` | Files still missing full required enrichment. |
| `fully_enriched` | `deep_pending == 0`. |
| `by_state` | Manifest state counts. |

## Item Metadata

Evidence items are typed retrieval units. `address_kind` identifies the unit:

| Kind | Typical source |
|---|---|
| `section` | Markdown, PDF, DOCX, PPTX, HTML, text, or parsed prose section. |
| `symbol` | Code function, class, method, constant, or module-level symbol. |
| `table` | CSV/table metadata or extracted document table. |
| `file` | Supplemental file-level match before full indexing. |

`excerpt` is display text. `content` is the fuller text passed into Pyrrho and
optional synthesis.
