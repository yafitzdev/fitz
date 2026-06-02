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
Pyrrho classifier (single local INT8 ONNX forward pass)
  │
  ▼
TRUSTWORTHY / DISPUTED / ABSTAIN
  │
  ▼
EvidencePack is returned; optional synthesizer can use the mode
```

The classifier is [`yafitzdev/pyrrho-nano-g3`](https://huggingface.co/yafitzdev/pyrrho-nano-g3)
on HuggingFace — a fine-tune of `answerdotai/ModernBERT-base` on the
fitz-gov V8.0.0 benchmark. The model card has the full headline numbers;
the short version:

| Metric                     | Pyrrho nano g3 |
| -------------------------- | -------------- |
| Held-out test split        | 2,459 examples, 3 seeds |
| Overall accuracy           | **97.52% ± 0.43** |
| ABSTAIN recall             | **97.83% ± 0.76** |
| DISPUTED recall            | **98.34% ± 0.24** |
| TRUSTWORTHY recall         | **96.28% ± 0.83** |
| False-trustworthy rate     | **1.42% ± 0.16** |
| External LLM dependency    | **none** |

---

## What changed in v0.13.0

The legacy constraint+sklearn cascade was removed entirely:

- 5 constraint plugins (`InsufficientEvidence`, `ConflictAware`,
  `CausalAttribution`, `SpecificInfoType`, `AnswerVerification`)
- `feature_extractor.py` (108-dim feature vector)
- `decider.py` (`GovernanceDecider` with the 4-question cascade
  classifier)
- `governor.py` (`AnswerGovernor`)
- `model_v6_cascade.joblib` (the trained sklearn artifact)
- `tools/governance/` (feature extraction + training scripts)

What remains in `fitz_sage/governance/`:

- `pyrrho.py` — the new ONNX inference module
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
```

`retrieved_contexts` is any sequence of objects satisfying the
`EvidenceItem` protocol — both `Chunk` and KRAG's `ReadResult`
qualify.

`create_governance("pyrrho")` returns a `Pyrrho` classifier instance.
The engine owns this instance and calls `.decide()` after retrieval and
before answer synthesis. The model is lazy-loaded on first decision;
that first call downloads/loads the ONNX export when missing, and
subsequent calls are a single local ONNX forward pass.

---

## Calibrated decision rule

Raw `argmax` over the 3-way softmax gives the predicted class.
Production uses a **threshold-calibrated fallback** on the
`TRUSTWORTHY` probability to favour the safer modes:

```python
if pred == TRUSTWORTHY and P(TRUSTWORTHY) < TAU:
    pred = argmax over (ABSTAIN, DISPUTED)
```

`TAU = 0.60` is the default. This is the rule that produces the
headline numbers above.

---

## Where it plugs in

The `FitzKragEngine` runs the pyrrho classifier after retrieval and reranking:

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

## Why a classifier and not constraints?

The old cascade was a chain of small models + LLM judges + hand-coded
heuristics over 108 features. It worked but was:

- **Slow.** Each constraint did its own LLM call; total ~500–2000 ms.
- **Brittle.** Constraint thresholds drifted with chat-model changes.
- **Opaque.** Hard to debug: which signal moved the needle?
- **Coupled to embeddings.** `SemanticMatcher` needed an embedder
  fitz-sage no longer ships (v0.12.0 dropped the embedding API).

A single fine-tuned classifier replaces all of that. It sees the
same `(query, contexts)` pair, decides in one forward pass, and ships
as a deterministic INT8 ONNX model with adjacent external-data files
downloaded from the Hub.

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

- [pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-nano-g3)
- [fitz-gov benchmark](https://github.com/yafitzdev/fitz-gov) — the evaluation dataset
- [pyrrho training code](https://github.com/yafitzdev/pyrrho)
- [`features/governance/governance-benchmarking.md`](features/governance/governance-benchmarking.md) — historical notes on the pre-v0.13.0 cascade
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where governance fits in the engine pipeline
