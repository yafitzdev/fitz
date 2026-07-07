# Epistemic Honesty

## Problem

Most RAG systems confidently answer questions even when the answer isn't in the documents:

- **Q:** "What was our Q4 revenue?" (docs only cover Q1-Q3)
- **Typical RAG:** "Q4 revenue was $2.5M" (hallucinated)
- **FitzKRAG:** "I cannot find Q4 revenue figures in the provided documents. The available financial data covers Q1-Q3 only."

The system cannot distinguish between "I have evidence" and "I'm making an educated guess."

## Solution: Pyrrho evidence governance

Every `(query, retrieved evidence prefix)` pair runs through the **pyrrho**
fine-tuned classifier
([`yafitzdev/pyrrho-v2-nano-g1`](https://huggingface.co/yafitzdev/pyrrho-v2-nano-g1),
a ModernBERT model trained on fitz-gov-v2 evidence data). A single local CPU forward pass
returns one of `SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT`:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents.
    The available financial data covers Q1-Q3 only."

   Verdict: INSUFFICIENT
```

## How It Works

### Prefix cutoff classification

Each decision is one local classifier call on CPU, with no external LLM
dependency. The default v2 model is evidence-conditioned; it does not project
pre-retrieval query heads.

For evidence retrieval, Pyrrho runs incrementally:

1. classify `query + top 1 evidence item`;
2. if evidence is insufficient, classify `query + top 2`;
3. continue until the evidence is sufficient, a dispute is stable, or the
   cutoff is reached.

| Case it catches              | Resulting mode | Example                                                                        |
| ---------------------------- | -------------- | ------------------------------------------------------------------------------ |
| Sources disagree             | `DISPUTED`     | "Document A says X, but Document B says Y. The sources disagree."              |
| Insufficient evidence        | `INSUFFICIENT` | "I cannot find information about X in the provided documents."                 |
| Sufficient supporting evidence | `SUFFICIENT` | Direct answer with citations.                                                  |

### Evidence Modes

Every `EvidencePack` includes a **mode** indicating confidence level:

- `SUFFICIENT` — Strong evidence supports downstream answering
- `DISPUTED` — Sources conflict; both views are presented
- `INSUFFICIENT` — Insufficient evidence; downstream systems should not answer

## Key Design Decisions

1. **Standard product path** - `governance: pyrrho` is the default and Pyrrho governance is mandatory.

2. **Retrieval-first classification** - Governance labels the evidence pack before any optional synthesis.

3. **Explicit modes** - The mode field is first-class in the Answer dataclass, not a hidden flag.

4. **Fail-safe defaults** - When in doubt, mark evidence insufficient. Better to return insufficient evidence than to invite hallucination.

5. **Transparent reasoning** - When evidence is insufficient or disputed, the system explains why.

## Configuration

Governance is selected by the `governance:` field:

```yaml
governance: pyrrho  # default local Pyrrho v2 classifier
# governance: pyrrho/<hf-model-id>  # Pyrrho package
```

## Files

- **Governance backend:** `fitz_sage/governance/pyrrho.py`
- **Answer modes:** `fitz_sage/core/answer_mode.py` (AnswerMode enum)
- **Evidence contract:** `fitz_sage/core/evidence.py`

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

Verdict: SUFFICIENT
```

**With the pyrrho classifier:**
```
Answer: Q4 sales declined by 15% compared to Q3. However, I cannot determine
the causal factors from the available data. The documents mention increased
competition and seasonal patterns, but these are correlations, not confirmed causes.

Verdict: INSUFFICIENT
```

## Dependencies

- Runs locally with `onnxruntime`, `transformers`, `huggingface-hub`, and `numpy`
- Uses no external LLM call for the governance decision
- Works without any chat provider because it governs retrieved evidence

## Related Features

- **Multi-Hop Reasoning** - Iterative retrieval can gather more evidence, reducing insufficient verdicts
- **Hierarchical RAG** - Corpus summaries help detect when information is genuinely missing
- **Aggregation Queries** - Comprehensive retrieval reduces false negatives
