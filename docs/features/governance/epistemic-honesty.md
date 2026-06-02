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
([`yafitzdev/pyrrho-nano-g3.1`](https://huggingface.co/yafitzdev/pyrrho-nano-g3.1),
a multitask ModernBERT-base model trained on the fitz-gov benchmark). A
single local CPU forward pass
returns one of `TRUSTWORTHY`, `DISPUTED`, or `ABSTAIN`:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents.
    The available financial data covers Q1-Q3 only."

   Mode: ABSTAIN
```

## How It Works

### Prefix cutoff classification

Pyrrho replaces the constraint+sklearn cascade that fitz-sage used
through v0.12.x. Each decision is one local classifier call on CPU, no
external LLM dependency. g3.1 also classifies the query contract before recall.

For evidence retrieval, Pyrrho runs incrementally:

1. classify `query + top 1 evidence item`;
2. if it abstains, classify `query + top 2`;
3. continue until the evidence is trustworthy, a dispute is stable, or the
   cutoff is reached.

| Case it catches              | Resulting mode | Example                                                                        |
| ---------------------------- | -------------- | ------------------------------------------------------------------------------ |
| Sources disagree             | `DISPUTED`     | "Document A says X, but Document B says Y. The sources disagree."              |
| Insufficient evidence        | `ABSTAIN`      | "I cannot find information about X in the provided documents."                 |
| Sufficient supporting evidence | `TRUSTWORTHY` | Direct answer with citations.                                                  |

### Evidence Modes

Every `EvidencePack` includes a **mode** indicating confidence level:

- `TRUSTWORTHY` — Strong evidence supports downstream answering
- `DISPUTED` — Sources conflict; both views are presented
- `ABSTAIN` — Insufficient evidence; downstream systems should not answer

## Key Design Decisions

1. **Standard product path** - `governance: pyrrho` is the default and Pyrrho governance is mandatory.

2. **Retrieval-first classification** - Governance labels the evidence pack before any optional synthesis.

3. **Explicit modes** - The mode field is first-class in the Answer dataclass, not a hidden flag.

4. **Fail-safe defaults** - When in doubt, ABSTAIN. Better to return insufficient evidence than to invite hallucination.

5. **Transparent reasoning** - When abstaining or disputing, the system explains why.

## Configuration

Governance is selected by the `governance:` field:

```yaml
governance: pyrrho  # default local Pyrrho g3.1 classifier
# governance: pyrrho/<hf-model-id>  # custom pyrrho fine-tune
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

- Runs locally with `onnxruntime`, `transformers`, `huggingface-hub`, and `numpy`
- Uses no external LLM call for the governance decision
- Works without any chat provider because it governs retrieved evidence

## Related Features

- **Multi-Hop Reasoning** - Iterative retrieval can gather more evidence, reducing ABSTAIN rate
- **Hierarchical RAG** - Corpus summaries help detect when information is genuinely missing
- **Aggregation Queries** - Comprehensive retrieval reduces false negatives
