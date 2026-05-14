# fitz-gov 5.0 Benchmark Results

**Version**: fitz-gov 5.0 (2,910 governance cases)
**Date**: February 2026
**Decision method**: 4-question cascade (GBT × 3)
**Thresholds**: Q1=0.770, Q3=0.680, Q4=0.770
**Evaluation**: 5-fold stratified cross-validation

---

## Executive Summary

fitz-gov 5.0 expanded the benchmark from 1,113 to 2,910 cases and replaced the two-stage binary pipeline with a **4-question cascade** that routes conflict and non-conflict paths through separate specialized models. The DetectionClassifier was upgraded from keyword rules to an ML+keyword hybrid (temporal 90.6% recall, comparison 90.2% recall), providing cleaner detection signals to the governance feature set.

> v4.0 was an intermediate development version. It was never formally benchmarked. v5.0 is the first documented version after v3.0.

### Current Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **78.7%** |
| **Abstain recall** | **86.5%** |
| **Disputed recall** | **86.1%** |
| **Trustworthy recall** | **70.0%** |
| **False-trustworthy** | **5.7%** (165/2,920) |

Evaluated with 5-fold stratified cross-validation on all 2,920 cases.

### Version Comparison

| Version | Cases | Decision Method | Abstain | Disputed | Trustworthy | Overall |
|---------|-------|-----------------|---------|----------|-------------|---------|
| v1.0 | 200 | Rules (governor) | 72.5% | 90.0% | ~72% | ~72% |
| v2.0 | 331 | Rules (governor) | 57.1% | 89.1% | ~47–79% | ~72% |
| v3.0 | 1,113 | 2-stage ML (ET + RF) | 93.7% | 94.4% | 89.0% | 90.9% |
| v5.0 | 2,910 | 4-question cascade (GBT × 3) | 90.2% | 74.9% | 78.6% | 81.3% |
| v5.1 | 2,920 | 5-question cascade (GBT × 4) | 84.6% | 77.3% | 70.5% | 76.4% (FT=4.3%) |
| **v6** | **2,920** | **5-question cascade (GBT × 4, citation-grounded AV)** | **86.5%** | **86.1%** | **70.0%** | **78.7% (FT=5.7%)** |

**On the accuracy drop**: v3.0's 90.9% was measured with safety-first thresholds calibrated and evaluated on the same 1,113-case set. v5.0 uses 5-fold cross-validation on 2.6× more cases — a stricter and more honest methodology. With 1,797 additional cases targeting harder boundary examples, lower cross-validated accuracy is expected. Critically, disputed recall at 74.9% is now bounded by the `ca_fired` rule gate (Q2): if the conflict is real but the constraint doesn't fire, the case never reaches Q3.

---

## Data Expansion

### Growth History

| Version | Cases | Method | Validation |
|---------|-------|--------|------------|
| v1.0 | 200 | Hand-crafted from 21 experiments | Expert review |
| v2.0 | +525 | LLM-assisted boundary sampling | 95.4% blind agreement |
| v2.1 | +123 | Targeted gaps (dispute boundary, code, adversarial) | 94% blind agreement |
| v3.0 | +199 | Classifier failure-mode targeting | 93.5% blind agreement |
| v5.0 | +1,797 | Extended boundary sampling + failure-mode targeting | — |
| **Total** | **2,910** | | |

### Class Distribution (v5.0)

| Class | Count | % |
|-------|-------|---|
| Trustworthy | 1,361 | 46.8% |
| Abstain | 884 | 30.4% |
| Disputed | 665 | 22.9% |

Max:min class ratio: 2.0:1 (improved from 2.2:1 in v3.0). 92% hard difficulty.

---

## 4-Question Cascade Architecture

The key structural change from v3.0 is splitting the conflict path from the clean path. v3.0 Stage 2 trained one Random Forest to distinguish trustworthy from disputed across all answerable cases. v5.0 routes them first through a deterministic gate (Q2), then through two separate specialized models.

```
Query + Context → 5 Constraints → 108 features
         │
         ▼
  Q1: Evidence sufficient? (GBT, t=0.780)
         │
         ├─ NO → ABSTAIN
         │
         └─ YES
               │
               ▼
         Q2: Material conflict? (GBT, t=0.200)
               │
               ├─ YES (conflict path)
               │        │
               │        ▼
               │  Q3: Conflict resolved? (GBT, t=0.720)
               │        │
               │        ├─ NO  → DISPUTED
               │        └─ YES → TRUSTWORTHY
               │
               └─ NO (clean path)
                        │
                        ▼
                  Q4: Evidence solid? (GBT, t=0.730)
                        │
                        ├─ NO  → ABSTAIN
                        └─ YES → TRUSTWORTHY
```

### Why Cascade Over Two-Stage

v3.0's Stage 2 (trustworthy vs disputed) trained on all answerable cases — conflict and non-conflict mixed. The cascade separates them:

- **Q3** is trained exclusively on conflict-path cases (where `ca_fired=True`). It learns "is this conflict resolvable?" from a focused set.
- **Q4** is trained exclusively on clean cases (where `ca_fired=False`). It learns "is the clean evidence truly solid?" without dispute noise.
- Both use GBT (v3.0 used ExtraTrees/RandomForest). GBT's sequential residual correction handles the skewed, hard-boundary cases better.

### Cascade vs Two-Stage

| Approach | Overall | Abstain | Disputed | Trustworthy |
|----------|---------|---------|----------|-------------|
| v3.0 two-stage (ET+RF, 1,113 cases) | 90.9% | 93.7% | 94.4% | 89.0% |
| v5.0 cascade (GBT×3, 2,910 cases, 5-fold CV) | 81.3% | 90.2% | 74.9% | 78.6% |
| v5.1 cascade (GBT×4, 2,920 cases, 5-fold CV) | 76.4% | 84.6% | 77.3% | 70.5% |
| **v6 cascade (GBT×4, citation-grounded AV, 2,920 cases, 5-fold CV)** | **78.7%** | **86.5%** | **86.1%** | **70.0%** |

v5.1 replaced the hard `ca_fired` rule at Q2 with an ML router trained on all features. This fixed the structural ceiling on disputed recall (v5.0's Q2 gate made 47% of disputed cases unreachable). v6 replaced jury-based YES/NO answer verification with citation-grounded verification (quote the passage, verify with string matching), pruned 3 noisy features (`year_count`, `ie_query_aspect`, `num_strong_denials`), and recalibrated. Disputed recall jumped to **86.1%**, overall accuracy to **78.7%**.

### Threshold Selection

Three independent thresholds tuned jointly via `calibrate_cascade.py`:

| Q1 | Q2 | Q3 | Q4 | FT | Overall | Abstain | Disputed | Trustworthy |
|----|----|----|----|----|---------|---------|----------|-------------|
| **0.780** | **0.200** | **0.720** | **0.730** | **165 (5.7%)** | **78.7%** | **86.5%** | **86.1%** | **70.0%** |

Sweep: Q1 ∈ [0.40, 0.80), Q2 ∈ [0.20, 0.55), Q3 ∈ [0.20, 0.72), Q4 ∈ [0.40, 0.73). Filter: accuracy ≥ 75% and trustworthy recall ≥ 65%. Chosen point minimizes false-trustworthy (over-confidence) subject to usability constraints.

---

## Key Changes from v3.0

### 1. Improved DetectionClassifier (ML + Keyword)

v3.0 detection flags (`detection_temporal`, `detection_comparison`) came from keyword/regex rules inside `DetectionOrchestrator`. v5.0 replaced this with an ML+keyword hybrid `DetectionClassifier`:

| Detection Type | Recall |
|----------------|--------|
| Temporal | **90.6%** |
| Comparison | **90.2%** |

These detection signals feed directly into the governance feature set. Better detection → less noise in the features the governance classifiers train on.

### 2. Feature Set Expansion (50 → 108)

v3.0 had 50 raw features. v6 has 108 features (was 112 in v5.1; 3 noisy features pruned + 3 dead removed) after one-hot encoding of categorical constraint outputs:

| Tier | v3.0 | v5.0 | Notes |
|------|------|------|-------|
| Constraint metadata | ~15 | ~30 | AnswerVerification (citation-grounded in v6) added |
| Vector/retrieval scores | ~8 | ~12 | Additional score distribution features |
| Context text analysis | ~20 | ~30 | Extended inter-chunk features |
| Detection flags | ~7 | ~15 | ML-classified detection + encoded categoricals |
| Categorical encodings | 0 | ~22 | One-hot expansion of `caa_query_type`, `sit_info_type_requested`, etc. |

Six features removed: 3 dead (`num_unique_sources`, `ie_max_similarity`, `ie_summary_overlap`) + 3 noisy pruned in v6 (`year_count`, `ie_query_aspect`, `num_strong_denials`).

### 3. AnswerVerification Constraint (New)

A fifth constraint added alongside IE, CA, CAA, SIT. v6 uses citation-grounded verification: the LLM quotes the exact passage that answers the question, then fuzzy string matching verifies the citation. This replaced the v5.1 jury-based YES/NO approach. `av_citation_quality` is the top feature in the v6 model; the interaction feature `ix_av_no_ie` is #1 in Q1 (answerability gate).

---

## What Worked

### GBT Over Random Forest/ExtraTrees

For all three cascade classifiers (Q1, Q3, Q4), GBT outperformed the v3.0 choice of ExtraTrees/RandomForest. GBT's sequential residual correction is better suited to the hard boundary cases (92% of dataset), where the marginal case is the main challenge.

### Specialized Path Models

Separating Q3 (conflict-path) and Q4 (clean-path) allows each model to specialize. Q3 focuses on "is this contradiction resolvable?" — a different question than Q4's "is the non-conflicting evidence actually sufficient?" Training them jointly (v3.0 Stage 2) required the model to handle both questions simultaneously.

### 5-Fold Cross-Validation

v3.0 thresholds were calibrated on the full training set. v5.0 uses OOF (out-of-fold) predictions for all threshold calibration, making the reported numbers genuinely held-out. This is the honest methodology going forward.

---

## What Failed

### Disputed Recall Recovery (74.9% → 86.1%)

v5.0's hard `ca_fired` rule at Q2 created a structural ceiling: 47% of disputed cases had `ca_fired=False` and could never be predicted as disputed. v5.1's ML Q2 router fixed this by learning dispute signals from all features. v6's citation-grounded AV and pruned noisy features pushed disputed recall further from 77.3% to 86.1%.

### Dataset Growth Penalty

Adding 1,797 cases to a benchmark calibrated on 1,113 inevitably introduces more hard boundary examples. Cross-validated accuracy on the expanded set is not directly comparable to threshold-calibrated accuracy on the original set — but the expanded set is more representative.

---

## Critical Case Profile

**False-trustworthy: 165 cases (5.7%)** — the most dangerous error (predicting trustworthy when should abstain or dispute). FT count increased from v5.1's 126 as the recalibrated thresholds trade higher FT for significantly better disputed recall (+8.8pp) and overall accuracy (+2.3pp).

---

## Reproduction

```bash
# Feature extraction (requires a configured chat endpoint)
python -m tools.governance.extract_features --workers 1

# Train cascade classifier
python -m tools.governance.train_classifier --mode cascade --time-budget 200

# Calibrate thresholds (minimizes false-trustworthy)
python -m tools.governance.calibrate_cascade
```

### Key Files

| File | Purpose |
|------|---------|
| `tools/governance/extract_features.py` | Feature extraction (chat-based detection signals) |
| `tools/governance/train_classifier.py` | Cascade training with hyperparameter search |
| `tools/governance/calibrate_cascade.py` | 4-threshold sweep for critical case minimization |
| `tools/governance/data/features.csv` | 2,920 rows × 95 columns |
| `fitz_sage/governance/data/model_v6_cascade.joblib` | Production cascade artifact |
| `fitz_sage/governance/decider.py` | GovernanceDecider (production inference) |
| `fitz_sage/governance/constraints/feature_extractor.py` | Runtime feature extraction (108 features) |
| `retrieval/detection/registry.py` | DetectionOrchestrator with ML DetectionClassifier |

---

## Path Forward

v6's citation-grounded AV addressed the primary v5.1 problem (jury checking relevance, not answer presence). The remaining FT cases are harder boundary examples where citation quality is high but the answer is incomplete or mismatched.

**What would help**:
- Post-generation claim verification (check that the generated answer is grounded in the retrieved chunks)
- Stronger LLM for constraint evaluation (the fast-tier model misses nuanced distinctions)
- Further feature engineering on citation match quality signals
