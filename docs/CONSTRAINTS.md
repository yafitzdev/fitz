# Epistemic Governance

How fitz-sage decides whether retrieved evidence is sufficient, disputed, or
insufficient.

---

## Overview

Most RAG systems confidently answer even when they shouldn't. fitz-sage
classifies `(query, retrieved evidence prefix)` pairs into one of three modes:

| Mode          | Meaning                                                                |
| ------------- | ---------------------------------------------------------------------- |
| `TRUSTWORTHY` | Sources consistently and sufficiently support an answer.               |
| `DISPUTED`    | Sources contradict each other on the answer.                           |
| `ABSTAIN`     | Sources do not contain enough information to answer.                   |

```
User query
  │
  ▼
Retrieve via KRAG + FTS5
  │
  ▼
Pyrrho g4-alpha classifier (local CPU forward pass)
  │
  ▼
TRUSTWORTHY / DISPUTED / ABSTAIN
  │
  ▼
EvidencePack is returned; optional synthesizer can use the mode
```

The classifier is
[`yafitzdev/pyrrho-nano-g4-alpha`](https://huggingface.co/yafitzdev/pyrrho-nano-g4-alpha)
on Hugging Face. It is a multitask `answerdotai/ModernBERT-base`
classifier trained from fitz-gov data plus query and evidence signal labels.
It exposes:

| Head | Purpose |
| ---- | ------- |
| Governance | `TRUSTWORTHY` / `DISPUTED` / `ABSTAIN` over `query + evidence prefix`. |
| Query contract | Pre-retrieval shape: evidence sufficiency, structured lookup, temporal grounding, exhaustive coverage, comparison coverage, representative overview. |
| Route/domain | Broad domain label for observability. |
| Taxonomy | Evidence-pattern label such as direct answer, conflict, missing evidence, wrong specificity. |
| Retrieval action | Evidence-conditioned action: answer now, retrieve more, broaden, resolve conflict, clarify, or structured lookup. |
| Gap type | Evidence gap label such as missing fact, missing timeframe, conflicting values, wrong entity, wrong scope, or unsupported inference. |
| Answerability shape | Query-only answer shape: direct, synthesis, set, or structured reasoning. |
| Retrieval modality | Query-only preferred evidence surface: text, table, code, config, logs, PDF layout, or mixed. |
| Scalars | Evidence sufficiency, alignment, coverage, conflict density, retry value, false-trustworthy risk, and failure severity. |

---

## Implementation

- `pyrrho.py` — the local Pyrrho g4-alpha inference module
- `protocol.py` — the `EvidenceItem` protocol (any object with
  `.content` + `.metadata`)
- `instructions.py` — the small `AnswerMode → prompt instruction` map

---

## Public API

```python
from fitz_sage.governance import GovernanceDecision, create_governance

governance = create_governance("pyrrho")
decision = governance.decide(query, retrieved_contexts)
# decision.mode    → AnswerMode (TRUSTWORTHY / DISPUTED / ABSTAIN)
# decision.probs   → (p_abstain, p_disputed, p_trustworthy)
# decision.reason  → one-line human-readable explanation
# decision exposes g4-alpha query, evidence, route, taxonomy, action, gap, and modality head metadata
# decision.scalars exposes the retrieval-relevant scalar heads
```

`retrieved_contexts` is any sequence of objects satisfying the
`EvidenceItem` protocol — both `Chunk` and KRAG's `ReadResult`
qualify.

`create_governance("pyrrho")` returns a `Pyrrho` classifier instance.
The engine owns this instance and calls `.decide()` after retrieval and
before answer synthesis. The model is lazy-loaded on first decision;
that first call downloads/loads the safetensors checkpoint when missing, and
subsequent calls are local CPU forward passes.

---

## Calibrated decision rule

Raw `argmax` over the 3-way softmax gives the predicted class.
Production uses a **threshold-calibrated fallback** on the
`TRUSTWORTHY` probability to favour the safer modes:

```python
if pred == TRUSTWORTHY and P(TRUSTWORTHY) < TAU:
    pred = argmax over (ABSTAIN, DISPUTED)
```

`TAU = 0.44` is the default for g4-alpha.

---

## Where it plugs in

The `FitzKragEngine` uses Pyrrho twice:

1. Before recall, Pyrrho classifies query signals. The query contract, route,
   answerability shape, and preferred retrieval modality steer recall profile
   and cutoff policy.
2. After reranking, Pyrrho evaluates evidence prefixes.

The cutoff loop:

1. Broad recall + rerank candidates → ranked `ReadResult`s.
2. Pyrrho scores `query + top 1`.
3. If the verdict is `ABSTAIN`, the engine adds the next evidence item and
   scores again.
4. The loop stops when evidence is `TRUSTWORTHY`, a dispute is stable, or the
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
local decision with mode probabilities, query-contract metadata, taxonomy
labels, and scalar risk signals. That keeps the cutoff loop fast and makes the
decision observable without adding endpoint calls to the retrieval path.

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
   false-trustworthy rate, so some answerable cases may be classified as
   `ABSTAIN` or `DISPUTED`.

---

## See Also

- [pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-nano-g4-alpha)
- [fitz-gov benchmark](https://github.com/yafitzdev/fitz-gov) — the evaluation dataset
- [pyrrho training code](https://github.com/yafitzdev/pyrrho)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where governance fits in the engine pipeline
