# Epistemic Honesty

## Problem

Retrieval can return useful-looking context even when the delivered sources do
not support an answer or disagree about it. Optional answer synthesis must not
silently treat every evidence set as sufficient.

## Pyrrho Governance

Every `(query, ranked evidence prefix)` runs through the local Pyrrho
classifier. Pyrrho returns:

- `SUFFICIENT`
- `DISPUTED`
- `INSUFFICIENT`

Fitz-Sage starts with up to three ranked items, adds two after exact
`INSUFFICIENT`, maps the native verdict name to `AnswerMode`, and includes the
stopping prefix plus exact serialized Pyrrho trajectory in the `EvidencePack`.

Pyrrho owns model loading, thresholds, label consistency, and the verdict.
Fitz-Sage does not add confidence safeguards or reinterpret the result with
query heuristics. Its only control rule is to continue the ranked prefix after
exact `INSUFFICIENT` and stop on exact `SUFFICIENT` or `DISPUTED`.

## Configuration

```yaml
governance: pyrrho
```

Bare `pyrrho` uses `yafitzdev/pyrrho-v2-nano-g1` at immutable revision
`948f0500b74871cfaec7689a01d4eab0dd516e1b`. A custom local package or remote
package pinned to a full 40-character commit can be selected explicitly.

## Runtime Boundary

- The decision is local CPU inference and does not use a chat model.
- Pyrrho sees only the evidence Fitz-Sage delivers.
- The accepted model currently has a 2,048-token input contract.
- False verdicts are Pyrrho model debt and should be addressed in Pyrrho
  training, not hidden by Fitz-Sage policy.
- Provider or package-contract failures are surfaced as errors; Fitz-Sage does
  not fabricate an `INSUFFICIENT` verdict.

Optional synthesis consumes the already-decided mode. It does not choose or
repair it.

See [Epistemic Governance](../../CONSTRAINTS.md) for the complete runtime
contract and [Limitations](../../LIMITATIONS.md) for measured behavior.
