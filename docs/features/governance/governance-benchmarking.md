# Governance Benchmarking (fitz-gov) — Historical (pre-v0.13.0)

> **Note.** This document describes the **constraint+sklearn cascade**
> that served as fitz-sage's governance backend through v0.12.x. As of
> v0.13.0 the cascade has been removed and replaced by the
> [pyrrho](https://huggingface.co/yafitzdev/pyrrho-nano-g3.1)
> classifier — see [`docs/CONSTRAINTS.md`](../../CONSTRAINTS.md) for
> the current architecture. The fitz-gov benchmark dataset itself is
> still active, and the current pyrrho-nano-g3.1 release reports
> 98.05% governance accuracy, 0.95% false-trustworthy, and 94.23%
> query-contract macro F1.

## The Problem with RAG Benchmarks

Standard RAG benchmarks (BEIR, MS MARCO, Natural Questions) measure **retrieval quality**: "Did you find the right documents?" Some newer benchmarks (RAGAS, TruLens) add **generation quality**: "Is the answer faithful to the retrieved context?"

Neither measures what actually matters for trust: **does the system know when it doesn't know?**

A RAG system that hallucinates confidently on every query can score perfectly on retrieval benchmarks. It found the right docs. It generated a fluent answer. The answer happens to be wrong because the docs contradict each other, but no benchmark caught that.

This is the gap fitz-gov fills.

## Why Epistemic Governance Is Harder Than Retrieval

Retrieval benchmarking has a clean signal: the document is relevant or it isn't. You can compute precision@k, recall@k, and nDCG against a gold standard. The problem is well-posed and the metrics are well-understood.

Epistemic governance has no clean signal. Consider:

- Two sources say different things. Is that a **dispute** (flag it) or a **resolved contradiction** (one supersedes the other, be confident)?
- Context is topically related but doesn't answer the specific question. Is that **abstain** (no answer possible) or **qualified** (partial answer with caveats)?
- Sources describe the same thing in different words. Is that **confident** (they agree) or **qualified** (they seem to disagree)?

These decisions require understanding *relationships between sources*, not just relevance to a query. A retrieval benchmark asks "is this document about the topic?" Governance asks "given these three documents that are all about the topic, should we trust the answer they produce?"

This is why the state of the art in governance accuracy is in the 60-75% range, while retrieval benchmarks routinely hit 90%+. The problem is fundamentally harder.

## fitz-gov: The Benchmark

[fitz-gov](https://github.com/yafitzdev/fitz-gov) is a governance calibration benchmark with **2,900+ labeled test cases** across 6 categories:

| Category | What It Tests | Cases | Example |
|------|---------------|-------|---------|
| **Abstention** | Context doesn't answer the question | 884 | Query: "Tokyo population" / Context: about Osaka |
| **Dispute** | Sources make mutually exclusive claims | 665 | Source A: "Project succeeded" / Source B: "Project failed" |
| **Trustworthy Hedged** | Evidence exists but needs caveats | ~700 | Causal question with only correlational data |
| **Trustworthy Direct** | Clear, consistent evidence | ~661 | Direct factual answer supported by all sources |

Each test case provides a query, 2-4 context passages (bypassing retrieval for controlled testing), and the expected governance mode with a rationale.

### Case Difficulty

The benchmark is deliberately hard. 92% of cases are rated "hard" — boundary cases where multiple governance modes could plausibly apply. This is by design: easy cases (query about cats, context about dogs = abstain) aren't useful for distinguishing systems. The hard cases are where governance matters.

54 subcategories cover the full taxonomy of epistemic challenges: implicit contradictions, temporal conflicts, statistical disagreements, resolved contradictions, partial evidence, wrong entities, cross-domain evidence gaps, and more.

### Data Lineage

The cases were generated in 5 rounds with independent blind validation at each stage:

| Round | Cases | Method | Validation |
|-------|-------|--------|------------|
| 1 | 200 | Hand-crafted from 21 experiments | Expert review |
| 2 | +525 | LLM-assisted boundary sampling | 95.4% blind agreement |
| 3 | +123 | Targeted edge cases (code, adversarial) | 94% blind agreement |
| 4 | +199 | Classifier failure-mode targeting | 93.5% blind agreement |
| 5 | +1,797 | Extended boundary sampling + failure-mode targeting | — |

Blind validation means an independent LLM labels each case without seeing the expected answer. Cases below 90% agreement are reviewed, relabeled, or removed.

## How Fitz Classifies Governance

Fitz uses a **5-question cascade**: epistemic constraints extract features, then four specialized GBT classifiers make the final decision.

### Layer 1: Constraint Pipeline (Feature Extraction)

Five constraints run on every query, each contributing diagnostic signals:

| Constraint | What It Detects | Key Features |
|------------|----------------|--------------|
| **InsufficientEvidence** | Context-query relevance gaps | Entity match, aspect match, summary overlap |
| **ConflictAware** | Pairwise contradictions between sources | Contradiction signal, evidence character, numerical variance |
| **CausalAttribution** | Causal/predictive/speculative query types | Query type, evidence presence |
| **SpecificInfoType** | Missing specific information | Info type requested, entity mismatch |
| **AnswerVerification** | Citation-grounded verification (quote + string match) | Citation quality, count |

Additionally, retrieval-side features (BM25 scores, score distributions), inter-chunk text features (pairwise overlap, assertion density, TF-IDF similarity), and query classification features (temporal, comparison, aggregation detection) feed into the classifier. Detection signals use the ML+keyword `DetectionClassifier` (temporal 90.6% recall, comparison 90.2% recall).

**Total: 108 features** (after one-hot encoding of categorical constraint outputs, 3 noisy features pruned in v6) across constraint metadata, retrieval scores, text analysis, and query classification.

### Layer 2: 5-Question Cascade (Decision)

A 5-question cascade routes conflict and non-conflict cases through separate specialized models.

```
Q1: Evidence sufficient? (GBT, t=0.780)
    "Is there enough evidence to answer at all?"
    → NO  → ABSTAIN
    → YES → Q2

Q2: Material conflict? (GBT, t=0.200)
    "Do features suggest a dispute exists?"
    → YES → Q3 (conflict path)
    → NO  → Q4 (clean path)

Q3: Conflict resolved? (GBT, t=0.720) — conflict path only
    "Can we resolve the detected contradiction?"
    → NO  → DISPUTED
    → YES → TRUSTWORTHY

Q4: Evidence solid? (GBT, t=0.730) — clean path only
    "Is the non-conflicting evidence truly sufficient?"
    → NO  → ABSTAIN
    → YES → TRUSTWORTHY
```

The TRUSTWORTHY output drives two response behaviors at runtime: if constraints fired during processing, the response is hedged; if no constraints fired, it answers directly.

Why 5-question cascade over two-stage:
- **Q3 and Q4 see cleaner signal.** Q3 is trained exclusively on conflict-path cases; Q4 exclusively on clean cases. v3.0 Stage 2 mixed both, requiring one model to answer two different questions simultaneously.
- **Q2 is an ML router, not a hard rule.** Previous versions used `ca_fired` as a hard gate, creating a structural ceiling on disputed recall (47% of disputed cases had `ca_fired=False`). The ML router uses all features — numerical divergence, contradiction markers, text signals — to detect disputes even when the LLM constraint misses them.
- **GBT outperforms ET/RF on hard boundary cases.** Sequential residual correction handles the 92%-hard-difficulty dataset better than the bagging ensemble used in v3.0.

The cascade artifact is ~5MB, runs in microseconds, and adds zero latency to the pipeline. It fails open — if anything goes wrong, it falls back to the rule-based governor.

## Current Results

**Production model: 5-question cascade (GBT × 4), safety-first thresholds. 5-fold cross-validated on 2,920 cases.**

| Decision | Meaning | Recall |
|----------|---------|--------|
| **ABSTAIN** | Evidence doesn't answer the question | **86.5%** |
| **DISPUTED** | Sources contradict each other | **86.1%** |
| **TRUSTWORTHY** | Consistent, sufficient evidence | **70.0%** |

**Overall accuracy: 78.7%** | **False-trustworthy: 5.7%** (165/2,920)

For comparison, the rule-based governor achieves 63.9% accuracy on the same test set.

### Safety-First Threshold Tuning

Four thresholds tuned jointly to minimize false-trustworthy predictions (over-confidence):

- **Q1=0.780** — answerability gate. Conservative: routes ambiguous cases to ABSTAIN rather than forward.
- **Q2=0.200** — conflict router. Low threshold: routes more cases to the conflict path to catch disputes. False routing is handled by Q3.
- **Q3=0.720** — conflict-resolution gate. Lower threshold allows more disputed cases through; Q3 is already on the conflict path.
- **Q4=0.730** — clean-path sufficiency gate. Conservative: if evidence isn't clearly solid, ABSTAIN.

The most common error is over-hedging: predicting ABSTAIN or DISPUTED when the answer is actually TRUSTWORTHY. This is annoying but harmless. The opposite (over-confidence) is dangerous but rare — **5.7% false-trustworthy rate** (165/2,920 cases).

### Training History

| Experiment | Dataset | Accuracy | Key Finding |
|------------|---------|----------|-------------|
| 1 (baseline) | 914, synthetic features | 57.4% | Constraint features alone insufficient |
| 2 (+context features) | 914, synthetic | 71.0% | Context features dominate importance |
| 3 (CA tuning) | 914, synthetic | 69.4% | Tighter CA prompts: +7pp disputed, -1.6pp overall |
| 4 (real features) | 914, real | 41.0% | Distribution shift — synthetic≠real features |
| 5 (retrained on real) | 914, real | 68.9% | 83% disputed recall, vector features now important |
| 6 (expanded data) | 1113, real | 69.1% | +199 cases, confident +14pp, disputed -16pp (harder cases) |
| 7 (dead code cleanup) | 1113, 29 features | 80.7% | Removed 18 dead features, no regression |
| 8 (inter-chunk features) | 1113, 51 features | 82.1% | TF-IDF similarity, assertion density, length CV |
| 9 (feature parity fix) | 1113, 50 features | **90.9%** | Fixed embedding distribution — v3.0 production result |
| 10 (cascade + expanded data) | 2910, 109 features, 5-fold CV | 81.3% | 4-question cascade, GBT×3, ML DetectionClassifier |
| 11 (ML Q2 router + constraint fixes) | 2920, 109 features, 5-fold CV | 76.4% (FT=4.3%) | 5-question cascade, GBT×4, full pairwise conflict detection, per-chunk verification, ML Q2 router replaces hard ca_fired rule |
| 12 (citation-grounded AV + feature pruning) | 2920, 108 features, 5-fold CV | **78.7%** (FT=5.7%) | Citation-grounded AV replaces jury YES/NO, 3 noisy features pruned, `av_citation_quality` is top feature, `ix_av_no_ie` is #1 in Q1 |

### Why These Numbers Are Honest

We could report higher numbers. Training on easier cases, using a smaller test set, or optimizing for overall accuracy at the expense of safety would all inflate the headline metric. We don't.

- **92% of test cases are hard** — boundary cases, not softballs
- **5-fold cross-validated** — all reported numbers are out-of-fold; no threshold calibration on the training set
- **We report per-class recall** — a system that gets 90% overall by ignoring the disputed class is worse than one that gets 78% with balanced performance
- **We report false-trustworthy rate** — the most dangerous error (confidently answering when it shouldn't). Our FT rate of 5.7% means 94.3% of trustworthy predictions are correct.
- **Safety-first calibration** — thresholds are chosen to minimize false-trustworthy, not to maximize overall accuracy

## Key Techniques

### 1. Pairwise Contradiction Detection
Instead of classifying each chunk independently, compare chunks pairwise:
```
"Do these two texts contradict each other about the question?"
```
This is critical because single-chunk classification can't see inter-source conflicts.

### 2. Critical Entity Matching
Years and numbered qualifiers must match exactly — high token overlap isn't enough:
```
Query: "2024 World Series" + Context: "2021 World Series" -> ABSTAIN
Query: "type 2 diabetes" + Context: "type 1 diabetes" -> ABSTAIN
```

### 3. Evidence Character Analysis
Classify the rhetorical stance of each source: "assertive" (states facts), "hedged" (uses qualifiers), or "mixed." Two assertive sources that disagree = dispute. One hedged source + one assertive = qualified.

### 4. Inter-Chunk Text Features
Deterministic features computed from chunk text without LLM calls: pairwise TF-IDF cosine similarity, assertion density, numerical variance, chunk length coefficient of variation. These features were introduced in v3.0 and improved Stage 2 accuracy by +10.5pp at the time.

### 5. Citation-Grounded Verification
The LLM quotes the exact passage that answers the question, then fuzzy string matching verifies the citation. If no valid citation is found, the case is flagged. `av_citation_quality` (best match score across chunks) is the top feature in the v6 model.

## Feature Importance

The top features the classifier relies on:

| Rank | Feature | Category | What It Captures |
|------|---------|----------|-----------------|
| 1 | Context length (mean) | Context | Proxy for evidence depth |
| 2 | Context length (total) | Context | Total evidence available |
| 3 | Context length (std) | Context | Asymmetry between sources |
| 4 | Retrieval score (mean) | Retrieval | BM25 retrieval confidence |
| 5 | Disputed signal | Constraint | CA constraint fired |
| 6 | Chunk length CV | Inter-chunk | Length variation (disputed chunks have higher variance) |
| 7 | Max pairwise overlap | Inter-chunk | Text similarity between sources |
| 8 | Query word count | Query | Query complexity |
| 9 | Number variance | Context | Numerical disagreement between sources |
| 10 | Pairwise similarity (mean) | Inter-chunk | TF-IDF cosine between sources |

Context and inter-chunk features dominate because they're available for every case, while constraint signals only fire on specific patterns. Improving constraint signal quality is the path to better accuracy.

## Running the Benchmark

_Note: the paths and commands below describe the pre-v0.13.0 constraint-cascade implementation and no longer exist in the current codebase._

```bash
# Feature extraction (requires a configured chat endpoint)
python -m tools.governance.extract_features --workers 1

# Train cascade classifier
python -m tools.governance.train_classifier --mode cascade --time-budget 200

# Calibrate thresholds (minimizes false-trustworthy)
python -m tools.governance.calibrate_cascade

# Output: fitz_sage/governance/data/model_v6_cascade.joblib
```

## Files

_Note: the paths and commands below describe the pre-v0.13.0 constraint-cascade implementation and no longer exist in the current codebase._

| File | Purpose |
|------|---------|
| **Test cases** | [fitz-gov](https://github.com/yafitzdev/fitz-gov) (2,900+ cases) |
| **Constraints** | `fitz_sage/governance/constraints/plugins/` (IE, CA, CAA, SIT, AV) |
| **Feature extraction** | `fitz_sage/governance/constraints/feature_extractor.py` (108 features) |
| **GovernanceDecider** | `fitz_sage/governance/decider.py` (4-question cascade) |
| **Governor fallback** | `fitz_sage/governance/governor.py` (AnswerGovernor rule-based) |
| **Training pipeline** | `tools/governance/train_classifier.py` |
| **Threshold calibration** | `tools/governance/calibrate_cascade.py` |
| **Evaluation pipeline** | `tools/governance/eval_pipeline.py` |
| **Model artifact** | `fitz_sage/governance/data/model_v6_cascade.joblib` |

## Technical Specification

For the full experimental record with training history, ablation results, and what worked/failed:

**[fitz-gov 3.0 results](../evaluation/fitz-gov-3.0-results.md)** — How we got from 26.9% (rules) to 90.9% (two-stage ML)

**[fitz-gov 5.0 results](../evaluation/fitz-gov-5.0-results.md)** — 5-question cascade, 2,920 cases, 78.7% accuracy, 5.7% FT (v6, 5-fold CV)

## Why This Matters

No other open-source RAG framework publishes governance accuracy numbers. Not because they can't — because the numbers would be embarrassing. Without explicit governance logic, a RAG system's epistemic behavior is determined by the LLM's tendencies: usually over-confident, occasionally hallucinating "I don't know" for no reason, never flagging source contradictions.

Fitz measures it, publishes the numbers, and works to improve them. That's the difference between claiming honesty and proving it.
