# Epistemic Governance

How fitz-sage decides when to answer, when to flag a dispute, and when to
abstain.

---

## Overview

Most RAG systems confidently answer even when they shouldn't. fitz-sage
classifies every `(query, retrieved contexts)` pair into one of three
modes before generating an answer:

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
Pyrrho classifier (single INT8 ONNX forward pass, ~30 ms CPU)
  │
  ▼
TRUSTWORTHY / DISPUTED / ABSTAIN
  │
  ▼
Synthesizer (chat call) generates answer with the right epistemic posture
```

The classifier is [`yafitzdev/pyrrho-modernbert-base-v1`](https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1)
on HuggingFace — a fine-tune of `answerdotai/ModernBERT-base` on the
fitz-gov v5.1 benchmark. The model card has the full headline numbers;
the short version:

| Metric                     | Pyrrho v1     | Pre-v0.13.0 cascade  | Δ        |
| -------------------------- | ------------- | -------------------- | -------- |
| Overall accuracy           | **86.13%**    | 78.7%                | +7.43 pp |
| False-trustworthy rate     | **5.27%**     | 5.7%                 | -0.43 pp |
| Wall-clock per decision    | **~30 ms**    | ~500–2000 ms (5 chat calls) | ~50x faster |
| External LLM dependency    | **none**      | required             | —        |

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
that first call costs the ONNX-load latency (~1–2 s on CPU), and
subsequent calls are ~30 ms.

---

## Calibrated decision rule

Raw `argmax` over the 3-way softmax gives the predicted class.
Production uses a **threshold-calibrated fallback** on the
`TRUSTWORTHY` probability to favour the safer modes:

```python
if pred == TRUSTWORTHY and P(TRUSTWORTHY) < TAU:
    pred = argmax over (ABSTAIN, DISPUTED)
```

`TAU = 0.50` is the default. This is the rule that produces the
headline numbers above.

---

## Where it plugs in

The `FitzKragEngine` runs the pyrrho classifier between retrieval and generation:

1. Retrieve + expand + rerank candidates → `expanded`
2. The classifier scores `(sanitized_query, expanded)` → `governance.mode`
3. The synthesizer receives `answer_mode` and prepends the matching
   instruction from `governance/instructions.py`:
   - `TRUSTWORTHY` → answer clearly and directly
   - `DISPUTED` → state the disagreement, don't pick a side
   - `ABSTAIN` → state evidence is insufficient
4. The engine builds `gap_context` (for ABSTAIN) and a simple
   conflict reason (for DISPUTED) to pass to the synthesizer.

Set the classifier with `governance: <spec> | null` in `FitzKragConfig`.
`governance: pyrrho` (default) runs it; `governance: null` disables
governance entirely — the smoke test uses `null` to measure raw
retrieval timing.

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
as a 1.35 GB HF model (INT8 ONNX, deterministic). The +7 pp accuracy
and -0.43 pp false-trustworthy delta are the validation.

---

## Limitations

The model card calls out two known failure modes:

1. **Multi-source convergence misclassified as DISPUTED.** When multiple
   authoritative sources agree on a fact with small numerical variation
   within tolerance (e.g. four climate agencies citing 1.09–1.20 °C of
   warming), the model occasionally classifies the case as `DISPUTED`.
   ~57% error rate on the `multi_source_convergence` fitz-gov
   subcategory (n=7). v2 will target this with augmentation.
2. **Short clean factual contexts trigger over-abstention.** A single
   sentence answering the question with no surrounding methodology
   can be classified as `ABSTAIN`. Training data was 62.7% hard
   tier1 cases (rich, methodological contexts) — the model under-fits
   the short-clean pattern. Production RAG chunks are typically
   tier1-like and largely unaffected.

---

## See Also

- [pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1)
- [fitz-gov benchmark](https://github.com/yafitzdev/fitz-gov) — the evaluation dataset
- [pyrrho training code](https://github.com/yafitzdev/pyrrho)
- [`features/governance/governance-benchmarking.md`](features/governance/governance-benchmarking.md) — historical notes on the pre-v0.13.0 cascade
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where governance fits in the engine pipeline
