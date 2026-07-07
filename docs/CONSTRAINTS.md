# Epistemic Governance

How fitz-sage decides whether retrieved evidence is sufficient, disputed, or
insufficient.

---

## Overview

Most RAG systems confidently answer even when they shouldn't. Pyrrho v2
classifies `(query, retrieved evidence prefix)` pairs into one of three native
evidence verdicts:

| Verdict | Meaning |
|---|---|
| `SUFFICIENT` | Sources consistently and sufficiently support an answer. |
| `DISPUTED` | Sources contradict each other on the answer. |
| `INSUFFICIENT` | Sources do not contain enough information to answer. |

```
User query
  │
  ▼
Retrieve via KRAG + FTS5
  │
  ▼
Pyrrho v2 classifier (local CPU forward pass)
  │
  ▼
SUFFICIENT / DISPUTED / INSUFFICIENT
  │
  ▼
EvidencePack is returned; optional synthesizer can use the mode
```

The classifier is
[`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1)
on Hugging Face. It is a `ModernBERT` classifier trained from fitz-gov-v2
evidence data. The default v2 package exposes only native v2 heads:

| Head | Purpose |
| ---- | ------- |
| `evidence_verdict` | `SUFFICIENT` / `DISPUTED` / `INSUFFICIENT` over `query + evidence prefix`. |
| `failure_mode` | Reason for insufficient or disputed evidence. |
| `retrieval_intents` | Multi-label evidence intent metadata. |
| `evidence_kinds` | Multi-label evidence-surface metadata. |

---

## Implementation

- `pyrrho.py` — the local Pyrrho v2 inference module
- `protocol.py` — the `EvidenceItem` protocol (any object with
  `.content` + `.metadata`)
- `instructions.py` — the small `AnswerMode → prompt instruction` map

---

## Public API

```python
from fitz_sage.governance import GovernanceDecision, create_governance

governance = create_governance("pyrrho")
decision = governance.decide(query, retrieved_contexts)
# decision.mode    → runtime AnswerMode (TRUSTWORTHY / DISPUTED / ABSTAIN)
# decision.probs   → (p_abstain, p_disputed, p_trustworthy)
# decision.reason  → one-line human-readable explanation
# decision exposes native v2 verdict, failure, retrieval-intent, and evidence-kind metadata
```

`retrieved_contexts` is any sequence of objects satisfying the
`EvidenceItem` protocol — both `Chunk` and KRAG's `ReadResult`
qualify.

`create_governance("pyrrho")` returns a `Pyrrho` classifier instance.
The engine owns this instance and calls `.decide()` after retrieval and
before answer synthesis. The model is lazy-loaded on first decision;
that first call downloads/loads the ONNX package when missing, and
subsequent calls are local CPU forward passes.

---

## Calibrated decision rule

Raw `argmax` over the 3-way v2 evidence-verdict softmax gives the predicted
class. Production uses a **threshold-calibrated fallback** on the `SUFFICIENT`
probability to favour the safer modes:

```python
if pred == SUFFICIENT and P(SUFFICIENT) < TAU:
    pred = argmax over (INSUFFICIENT, DISPUTED)
```

`TAU = 0.34` is the default runtime threshold for v2.

---

## Where it plugs in

The `FitzKragEngine` uses the default Pyrrho v2 package after reranking to
evaluate evidence prefixes. Pre-retrieval Pyrrho query planning is inactive for
v2 until a query-trained v2 package exists.

The cutoff loop:

1. Broad recall + rerank candidates → ranked `ReadResult`s.
2. Pyrrho scores `query + top 1`.
3. If the verdict is `INSUFFICIENT`, the engine adds the next evidence item and
   scores again.
4. The loop stops when evidence is `SUFFICIENT`, a dispute is stable, or the
   cutoff is reached.
5. `evidence()` returns an `EvidencePack` with mode, reasons, probabilities,
   cutoff metadata, and source items.
6. Optional `answer()` synthesis receives the same mode and evidence.

`governance: pyrrho` is the product default and governance is mandatory in the
standard retrieval pipeline.

---

## Why a classifier?

Governance has to judge the whole `(query, evidence prefix)` pair, not just
individual keywords or source counts. Pyrrho gives the engine one calibrated
local verdict with mode probabilities and native v2 metadata. That keeps the
cutoff loop fast and makes the decision observable without adding endpoint calls
to the retrieval path.

---

## Limitations

The model card calls out these known boundaries:

1. **English-only training and evaluation data.**
2. **Source-bounded judgment.** Pyrrho judges only the retrieved evidence;
   it does not retrieve new evidence or verify claims against outside
   knowledge.
3. **Numeric agreement is learned, not hard-coded.** Exact numeric workflows
   should still be evaluated before deployment.
4. **Safety-tuned thresholding.** The decision threshold is tuned for low
   false-sufficient rate, so some answerable cases may be classified as
   `INSUFFICIENT` or `DISPUTED`.

---

## See Also

- [pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1)
- [fitz-gov on Hugging Face](https://huggingface.co/datasets/yafitzdev/fitz-gov-v2) — the evaluation dataset
- [pyrrho training code](https://github.com/yafitzdev/pyrrho)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where governance fits in the engine pipeline
