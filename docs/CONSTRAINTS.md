# Epistemic Governance

How fitz-sage decides whether retrieved evidence is sufficient, disputed, or
insufficient.

> **Quarantine notice (2026-07-15):** the historical default remote model
> `yafitzdev/pyrrho-v2-nano-g1` is blocked by default because its training
> corpus contained 5,000 benchmark-derived deterministic rows. Normal runtime
> use must supply an explicit local clean model directory. Forensic reproduction
> of the compromised, commit-pinned artifact requires the deliberate
> `FITZ_ALLOW_COMPROMISED_PYRRHO=1` escape hatch and is fixed to known-bad
> revision `948f0500b74871cfaec7689a01d4eab0dd516e1b`; it is not a production
> opt-in.

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

The historical classifier package is
[`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1)
on Hugging Face. It is preserved for forensic reproducibility and is not an
approved runtime default. A clean local v2 package exposes only native v2 heads:

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

governance = create_governance("pyrrho/C:/reviewed/clean/pyrrho-package")
decision = governance.decide(query, retrieved_contexts)
# decision.mode    → runtime AnswerMode (SUFFICIENT / DISPUTED / INSUFFICIENT)
# decision.probs   → (p_insufficient, p_disputed, p_sufficient)
# decision.reason  → one-line human-readable explanation
# decision exposes native v2 verdict, failure, retrieval-intent, and evidence-kind metadata
```

`retrieved_contexts` is any sequence of objects satisfying the
`EvidenceItem` protocol. KRAG's `ReadResult` qualifies, and custom engines can
provide their own compatible evidence type.

The bare `create_governance("pyrrho")` form currently fails closed because the
historical remote default is quarantined. Pass an explicitly reviewed local
package path as shown above. The engine owns the classifier and calls
`.decide()` after retrieval and before answer synthesis; the package is
lazy-loaded on the first decision and subsequent calls are local CPU forward
passes.

---

## Runtime threshold rule

Raw `argmax` over the 3-way v2 evidence-verdict softmax gives the predicted
class. The runtime applies a `SUFFICIENT` probability floor:

```python
if pred == SUFFICIENT and P(SUFFICIENT) < TAU:
    pred = argmax over (INSUFFICIENT, DISPUTED)
```

`TAU = 0.34` is the legacy fallback when a package does not declare
`release.sufficient_threshold`. It must not be described as calibrated for a
new clean model until runtime-exact threshold metrics are reported.

The verdict and failure heads are decoded independently, but the corpus
ontology permits only these pairs:

| Verdict | Valid failure mode |
|---|---|
| `SUFFICIENT` | `none` |
| `DISPUTED` | `unresolved_conflict` |
| `INSUFFICIENT` | `missing_or_incomplete_evidence`, `wrong_scope_or_version`, or `ambiguous_request` |

If independent predictions violate this matrix, the runtime reconciles them
without upgrading safety. It records the original pair and consistency fallback
in governance metadata.

Inputs are right-truncated at the manifest-declared token budget. Every post-
retrieval decision and pre-retrieval plan records the original token count,
budget, and whether truncation occurred; callers should treat truncation as an
observed evidence-coverage limitation.

---

## Where it plugs in

The `FitzKragEngine` profiles the query, retrieves and reranks candidates, then
uses the configured reviewed local Pyrrho v2 package to evaluate evidence
prefixes.

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

Governance is mandatory in the standard retrieval pipeline, but the bare
`governance: pyrrho` remote resolution is blocked until a clean release is
approved. Configure `governance: pyrrho/<local-package-path>`.

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
4. **Threshold calibration pending.** The clean release needs runtime-exact
   threshold evaluation; the legacy `0.34` fallback is not a new-release claim.
5. **Finite context.** Long evidence prefixes are right-truncated. Runtime
   metadata exposes this, but evidence beyond the token budget is unseen.

---

## See Also

- [quarantined historical pyrrho model card](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1)
- [quarantined published fitz-gov snapshot](https://huggingface.co/datasets/yafitzdev/fitz-gov-v2)
- [pyrrho training code](https://github.com/yafitzdev/pyrrho)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where governance fits in the engine pipeline
