# Multi-Hop Reasoning

## Problem

Some questions can't be answered from a single retrieval pass — the
answer is one reference away:

- **Q:** "Who wrote the paper cited by the 2023 review?"
- **Single-pass retrieval:** returns the 2023 review only. It contains
  the citation, not the cited paper or its author.
- **What's needed:** find the review → extract the citation → retrieve
  the cited paper → read off the author.

Following a chain like that needs **iterative retrieval**.

## Solution: loop the retrieval pass

`KragHopController` loops the **retrieval pass** — one round of
retrieve → rerank → read. After each pass it asks the pyrrho governance
classifier whether the accumulated evidence is enough:

```
Query → RetrievalPass (retrieve → rerank → read) → pyrrho verdict
          TRUSTWORTHY / DISPUTED  → stop — generate the answer
          ABSTAIN                 → extract a bridge question, loop
```

- `TRUSTWORTHY` / `DISPUTED` → **stop.** The evidence answers the
  question, or the sources disagree and more retrieval won't resolve it.
- `ABSTAIN` → **keep going.** Extract a bridge question from what's been
  read, and run another pass with it.

The sufficiency check is the pyrrho verdict — a ~30 ms INT8 ONNX forward
pass, **no chat call**. The only chat call multi-hop adds is bridge
extraction, spent only on the `ABSTAIN` path.

## On by default

`enable_multi_hop` defaults to `true`. A single hop is the common case:
for most queries pyrrho returns `TRUSTWORTHY` after the first pass and
the loop exits. The cost on that path is one ~30 ms pyrrho call for the
sufficiency check — no chat call. `max_hops` (default `2`) caps the loop.

## Key design decisions

1. **Pyrrho is the sufficiency signal.** The loop reuses the governance
   verdict instead of a separate "is this enough?" chat call — so
   multi-hop adds no chat call on the common single-hop path.
2. **Rerank inside every pass.** Each hop runs a full `RetrievalPass`,
   so the cross-encoder reranks every hop's candidates.
3. **Bridge extraction stays a chat call.** Writing a focused follow-up
   query is genuine text generation; it runs only when pyrrho `ABSTAIN`s.
4. **Deduplicated across hops.** Each pass skips the addresses earlier
   hops already read.
5. **Graceful stop.** The loop ends on a `TRUSTWORTHY` / `DISPUTED`
   verdict, an empty pass, no bridge question, or `max_hops`.

## Configuration

```yaml
enable_multi_hop: true   # default — loop the pass, pyrrho-gated
max_hops: 2              # default — hop cap (1-5)
```

`enable_multi_hop: false` runs a single retrieval pass with no loop.

## Files

| Component      | Path                                                      |
| -------------- | --------------------------------------------------------- |
| Hop controller | `fitz_sage/engines/fitz_krag/retrieval/multihop.py`       |
| Retrieval pass | `fitz_sage/engines/fitz_krag/retrieval/retrieval_pass.py` |
| Governance     | `fitz_sage/governance/pyrrho.py`                          |

## Example: citation chasing

**Query:** "Who wrote the paper cited by the 2023 review?"

- **Hop 1** — retrieve the 2023 review. Pyrrho `ABSTAIN`: the review
  cites the paper but doesn't name its author. Bridge question:
  *"Find the paper Smith et al. cited in the 2023 review."*
- **Hop 2** — retrieve the cited paper. Pyrrho `TRUSTWORTHY`: the author
  is in the evidence. Stop and answer.

## Related

- [Reranking](reranking.md) — the cross-encoder inside each pass
- [Epistemic Governance (pyrrho)](../../CONSTRAINTS.md) — the verdict that gates the loop
- [Comparison Queries](comparison-queries.md) — multi-entity retrieval (single-hop)
- [Temporal Queries](temporal-queries.md) — period filtering (single-hop)
