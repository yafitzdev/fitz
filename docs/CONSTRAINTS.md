# Epistemic Governance

How fitz-sage reports whether the delivered evidence is sufficient, disputed,
or insufficient.

## Overview

Pyrrho classifies one `(query, delivered evidence set)` pair into a native
evidence verdict:

| Verdict | Meaning |
|---|---|
| `SUFFICIENT` | The delivered sources consistently and sufficiently support an answer. |
| `DISPUTED` | The delivered sources contradict each other on the answer. |
| `INSUFFICIENT` | The delivered sources do not contain enough information to answer. |

```text
User query
  -> Fitz retrieval, reranking, closure, and compilation
  -> fixed delivered evidence set
  -> one local Pyrrho decision
  -> EvidencePack with exact Pyrrho metadata
```

## Ownership

- The Pyrrho ONNX model supplies the learned governance judgment. Fitz-Sage's
  managed adapter owns model resolution, artifact validation, tokenization,
  and mechanical head decoding.
- Fitz-Sage owns query-shape recognition, retrieval, evidence closure,
  compilation, and delivery.
- `fitz_sage/integrations/pyrrho.py` sends source IDs plus unchanged source
  text, maps the returned verdict name to `AnswerMode`, and serializes the exact
  Pyrrho decision.
- Fitz-Sage does not apply a second confidence floor, conflict heuristic,
  evidence-count rule, retry loop, or query-shape override to the verdict.

## Default Model

Bare `pyrrho` resolves the CPU-local ONNX model
[`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1)
at immutable revision:

```text
948f0500b74871cfaec7689a01d4eab0dd516e1b
```

Fitz-Sage downloads it lazily into the Hugging Face cache. A local model
directory or another remote model pinned to a full 40-character
commit can be configured explicitly.

```yaml
governance: pyrrho
# governance: pyrrho/C:/models/custom-pyrrho
# governance: pyrrho/owner/repository@0123456789abcdef0123456789abcdef01234567
```

## Model Contract

Pyrrho exposes four native heads:

| Head | Purpose |
|---|---|
| `evidence_verdict` | `SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT` |
| `failure_mode` | Reason for insufficient or disputed evidence |
| `retrieval_intents` | Multi-label query/evidence intent metadata |
| `evidence_kinds` | Multi-label evidence-surface metadata |

The managed adapter validates model manifests, label order, ONNX output width,
token limits, graph parity metadata, and verdict/failure compatibility. These
checks implement the model's declared interface; retrieval does not add a
second governance policy.

The accepted model currently has a 2,048-token input contract. Pyrrho records
the original token count, budget, and truncation status. Evidence beyond that
budget is unseen by the classifier even though Fitz-Sage can still return it in
the `EvidencePack`.

## Public API

```python
from fitz_sage.integrations.pyrrho import create_pyrrho

pyrrho = create_pyrrho("pyrrho")
decision = pyrrho.decide(query, retrieved_contexts)

print(decision.verdict)
print(decision.probabilities)
print(decision.to_dict())
```

The engine lazy-loads Pyrrho on the first query-plan or evidence-decision call.
After model download and initialization, decisions are local CPU inference and
do not require a chat model.

## Pipeline Contract

1. Fitz-Sage profiles the query and retrieves candidate evidence.
2. Reranking, closure, and compilation produce the final ranking.
3. `top_k`, or configured `top_read`, fixes the delivered evidence set.
4. Pyrrho evaluates the query plus exactly those evidence items once.
5. Fitz-Sage returns the same items and the exact serialized decision.

Optional synthesis consumes the already-decided mode. It does not choose or
repair that mode.

## Known Boundaries

1. Pyrrho's current model quality is accepted, not treated as solved. False
   sufficient, disputed, or insufficient decisions are Pyrrho model debt.
2. The current model and evaluation are English-focused.
3. Pyrrho judges only delivered evidence. It does not retrieve more material or
   verify claims against outside knowledge.
4. Numeric agreement and conflict recognition are learned rather than
   hard-coded.
5. The 2,048-token contract can hide evidence late in a large delivered set.

See [Limitations](LIMITATIONS.md) for the measured Fitz-Sage retrieval
boundary and the exact separation between retrieval and Pyrrho outcomes.
