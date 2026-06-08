<!-- docs/EVIDENCE_PACK.md -->
# Evidence Pack

`EvidencePack` is the retrieval-first response contract. It is returned by:

- `fitz query`
- `fitz retrieve`
- `fitz_sage.evidence()`

It is intentionally not an answer. It is ranked, governed evidence that another
application can inspect, display, or pass into optional synthesis.

For the meaning and product use of pre-retrieval and post-retrieval signals, see
[Pre-Retrieval and Post-Retrieval Evidence Signals](features/retrieval/evidence-signals.md).

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

| Mode | Meaning |
|---|---|
| `trustworthy` | Pyrrho judged the selected evidence prefix sufficient and consistent. |
| `disputed` | Pyrrho found meaningful conflict in the selected evidence prefix. |
| `abstain` | Retrieved evidence was missing, incomplete, or insufficient. |
| `null` | Governance did not run. This is not the default product path. |

## Governance Metadata

Pre-retrieval planning metadata lives in `metadata.query_profile`. It records
what Pyrrho predicted from the query alone and which profile knobs Fitz applied
before recall.

```json
{
  "metadata": {
    "query_profile": {
      "signals": {
        "query_contract": {
          "final_label": "comparison_coverage",
          "confidence": 0.97,
          "used_for_retrieval": true
        },
        "retrieval_modality": {
          "final_label": "structured_table",
          "confidence": 0.61,
          "used_for_retrieval": true
        }
      },
      "profile": {
        "specificity": "moderate",
        "answer_type": "comparative",
        "domain": "technical",
        "top_k": 20,
        "top_read": 12,
        "strategy_weights": {
          "code": 0.25,
          "section": 0.25,
          "table": 0.55
        }
      }
    }
  }
}
```

Pyrrho cutoff metadata lives in `metadata.governance_cutoff`.

```json
{
  "metadata": {
    "governance_cutoff": {
      "evaluated": 3,
      "selected": 3,
      "max": 10,
      "mode": "trustworthy",
      "policy": {
        "query_shape": "broad",
        "min_trustworthy_docs": 4,
        "min_disputed_docs": 2,
        "disputed_patience_docs": 2
      },
      "pyrrho": {
        "mode": "trustworthy",
        "probabilities": {
          "abstain": 0.08,
          "disputed": 0.12,
          "trustworthy": 0.80
        },
        "reason": "Pyrrho: sources support a confident answer (P=0.80)."
      }
    }
  }
}
```

Retrieval trace metadata lives in `metadata.retrieval_trace`. It is always
included so benchmark and analysis tools can inspect how a pack was produced:

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

Evidence compiler metadata lives in `metadata.evidence_compiler`. It records the
Pyrrho query contract projected into retrieval, literal anchors used for
mechanical evidence matching, how many evidence items entered and left
compilation, the minimum source count required before governance may stop, and
the selected evidence roles:

```json
{
  "metadata": {
    "evidence_compiler": {
      "contract": {
        "query_contract": "temporal_grounding",
        "route": "technology_computing",
        "answerability_shape": "direct_answer",
        "retrieval_modality": "structured_table",
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

Field meanings:

| Field | Meaning |
|---|---|
| `evaluated` | How many ranked evidence prefixes Pyrrho evaluated. |
| `selected` | How many evidence items were returned after cutoff. |
| `max` | Maximum cutoff window for this query, capped at 10 by default. |
| `mode` | Final governance mode for the selected prefix. |
| `policy.query_shape` | Narrow, broad, comparison, or aggregation. |
| `policy.min_trustworthy_docs` | Minimum prefix size before `TRUSTWORTHY` can stop. |
| `policy.min_disputed_docs` | Minimum prefix size before comparison disputes can stop. |
| `policy.disputed_patience_docs` | Additional patience for narrow disputes. |
| `pyrrho.probabilities` | Softmax probabilities for abstain, disputed, trustworthy. |
| `pyrrho.reason` | Human-readable one-line explanation. |
| `query_profile.signals.*.used_for_retrieval` | Whether the signal passed the confidence guard and changed the retrieval profile. |
| `query_profile.profile` | The effective pre-retrieval profile knobs used by recall. |

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
