# Epistemic Honesty

## Problem

Most RAG systems confidently answer questions even when the answer isn't in the documents:

- **Q:** "What was our Q4 revenue?" (docs only cover Q1-Q3)
- **Typical RAG:** "Q4 revenue was $2.5M" (hallucinated)
- **FitzKRAG:** "I cannot find Q4 revenue figures in the provided documents. The available financial data covers Q1-Q3 only."

The system cannot distinguish between "I have evidence" and "I'm making an educated guess."

## Solution: pyrrho classifier

Every `(query, retrieved contexts)` pair runs through the **pyrrho**
fine-tuned classifier (a ModernBERT-base head distilled on the
fitz-gov benchmark, served as INT8 ONNX). A single forward pass
returns one of `TRUSTWORTHY`, `DISPUTED`, or `ABSTAIN`:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents.
    The available financial data covers Q1-Q3 only."

   Mode: ABSTAIN
```

## How It Works

### Single-pass classification

Pyrrho replaces the constraint+sklearn cascade that fitz-sage used
through v0.12.x. Each decision is one ONNX inference call (~30 ms
on CPU), no external LLM dependency.

| Case it catches              | Resulting mode | Example                                                                        |
| ---------------------------- | -------------- | ------------------------------------------------------------------------------ |
| Sources disagree             | `DISPUTED`     | "Document A says X, but Document B says Y. The sources disagree."              |
| Insufficient evidence        | `ABSTAIN`      | "I cannot find information about X in the provided documents."                 |
| Sufficient supporting evidence | `TRUSTWORTHY` | Direct answer with citations.                                                  |

### Answer Modes

Every answer includes a **mode** indicating confidence level:

- `TRUSTWORTHY` — Strong evidence supports the answer across multiple sources
- `DISPUTED` — Sources conflict; both views are presented
- `ABSTAIN` — Insufficient evidence; refuses to answer

## Key Design Decisions

1. **Always-on** - The pyrrho classifier runs automatically on every answer. No configuration needed.

2. **Post-generation filtering** - The governance classifier evaluates the LLM's answer and retrieved chunks, not the raw query.

3. **Explicit modes** - The mode field is first-class in the Answer dataclass, not a hidden flag.

4. **Fail-safe defaults** - When in doubt, ABSTAIN. Better to say "I don't know" than to hallucinate.

5. **Transparent reasoning** - When abstaining or disputing, the system explains why.

## Configuration

No configuration required. The pyrrho classifier is baked into the answer generation pipeline. Governance is selected by the `governance:` field (`pyrrho` or `null`).

## Files

- **Governance backend:** `fitz_sage/governance/pyrrho.py`
- **Answer modes:** `fitz_sage/core/answer_mode.py` (AnswerMode enum)

## Benefits

| Without Epistemic Honesty | With Epistemic Honesty |
|---------------------------|------------------------|
| Hallucinated answers look confident | "I don't know" when uncertain |
| No way to detect conflicts | Surfaces contradictions explicitly |
| Users can't trust output | Transparent confidence signaling |
| Dangerous for high-stakes domains | Safe for compliance, legal, medical |

## Example

**Query:** "What caused the Q4 sales decline?"

**Without the governance classifier:**
```
Answer: The Q4 sales decline was primarily caused by increased competition
and seasonal factors.

Mode: TRUSTWORTHY
```

**With the pyrrho classifier:**
```
Answer: Q4 sales declined by 15% compared to Q3. However, I cannot determine
the causal factors from the available data. The documents mention increased
competition and seasonal patterns, but these are correlations, not confirmed causes.

Mode: TRUSTWORTHY
```

## Dependencies

- No external dependencies
- Pure Python implementation
- Works with any LLM provider

## Related Features

- **Multi-Hop Reasoning** - Iterative retrieval can gather more evidence, reducing ABSTAIN rate
- **Hierarchical RAG** - Corpus summaries help detect when information is genuinely missing
- **Aggregation Queries** - Comprehensive retrieval reduces false negatives
