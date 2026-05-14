# Epistemic Constraints

How Fitz knows when to say "I don't know."

---

## Overview

Most RAG systems confidently answer even when they shouldn't. Fitz uses **epistemic constraints** to detect problematic situations and respond appropriately.

```
┌─────────────────────────────────────────────────────────────────┐
│  User Query                                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Retrieve Chunks                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Apply Constraints                                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │
│  │ Insufficient  │ │ Conflict      │ │ Causal Attribution    │  │
│  │ Evidence      │ │ Aware         │ │                       │  │
│  └───────────────┘ └───────────────┘ └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Generate Answer with Appropriate Mode                          │
│  TRUSTWORTHY | DISPUTED | ABSTAIN                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Answer Modes

Constraints determine the answer mode (3-class system):

| Mode | Signal | User sees |
|------|--------|-----------|
| **TRUSTWORTHY** | No constraints triggered | Direct answer |
| **DISPUTED** | Sources conflict | "Sources disagree..." |
| **ABSTAIN** | Insufficient evidence | "I cannot find..." |

---

## Built-in Constraints

### 1. Insufficient Evidence

**Plugin:** `insufficient_evidence`

Prevents confident answers when there's not enough direct evidence.

**What it checks:**
- Are there any chunks retrieved?
- For causal questions: is there causal language in sources?
- For fact questions: are there direct assertions?

**Example:**

```
Q: "What was our Q4 revenue?"
Chunks: [Q1 revenue: $10M, Q2 revenue: $12M, Q3 revenue: $11M]

Constraint: Triggered - no Q4 data found
Mode: ABSTAIN
Answer: "I cannot find Q4 revenue figures in the provided documents.
         The available data covers Q1-Q3 only."
```

**Configuration:**

```yaml
# Enabled by default, no config needed
```

---

### 2. Conflict Aware

**Plugin:** `conflict_aware`

Detects when sources contradict each other.

**What it detects:**
- Mutually exclusive claims (A vs not-A)
- Contradicting values (increased vs decreased)
- Timeline conflicts (before vs after)

**Example:**

```
Q: "Was the 2023 incident a security breach?"
Chunk 1: "The 2023 incident was classified as a security breach..."
Chunk 2: "Investigation concluded this was an operational issue, not security..."

Constraint: Triggered - conflicting classifications
Mode: DISPUTED
Answer: "Sources disagree on this classification.
         - [Source 1] describes it as a security breach
         - [Source 2] classifies it as an operational issue"
```

**Configuration:**

```yaml
# Enabled by default, no config needed
```

---

### 3. Causal Attribution

**Plugin:** `causal_attribution`

Prevents inventing causal explanations not stated in sources.

**What it enforces:**
- Causal queries ("why", "what caused") need explicit causal language
- Won't synthesize causality from correlation
- Won't invent reasons not in the documents

**Example:**

```
Q: "Why did the deployment fail?"
Chunks: [Deployment failed at 3pm. Logs show timeout errors. System was restored at 5pm.]

Constraint: Triggered - no explicit "because" or causal language
Mode: ABSTAIN
Answer: "I can describe what happened but cannot determine the cause from
         the available documents. The deployment failed at 3pm with timeout
         errors. The documents don't explain why this occurred."
```

**Configuration:**

```yaml
# Enabled by default, no config needed
```

---

## How Constraints Work

### Constraint Result

Each constraint returns a result:

```python
ConstraintResult(
    allow_decisive_answer: bool,   # Can we give a confident answer?
    reason: str,                   # Human-readable explanation
    signal: str,                   # "abstain", "disputed", "qualified"
    metadata: dict                 # Debug info
)
```

### Execution flow

1. Retrieve relevant chunks
2. Run all constraints on (query, chunks)
3. Collect signals from any triggered constraints
4. Resolve to answer mode
5. Generate answer with appropriate framing

### Signal resolution

Constraint signals are used as **features** for a 5-question cascade ML classifier (GovernanceDecider), not as direct priority rules. The classifier predicts one of 3 classes:

| Classifier Output | AnswerMode | Condition |
|-------------------|------------|-----------|
| abstain | ABSTAIN | P(answerable) < threshold |
| disputed | DISPUTED | P(trustworthy) < threshold |
| trustworthy | TRUSTWORTHY | Evidence supports answer |

The AnswerGovernor still exists as a fallback when the GovernanceDecider model is not available, but GovernanceDecider is the primary decision path.

See [governance benchmarking](features/governance/governance-benchmarking.md) for details on the cascade classifier (78.7% accuracy, 5.7% false-trustworthy rate).

---

## How constraints detect signals

Constraints detect signals (conflict, missing evidence, causal claim
without support) by a mix of:

1. **Chat-LLM reasoning** — for the heavyweight checks
   (`conflict_aware`, `insufficient_evidence`, `answer_verification`)
   the constraint asks the `chat_fast` or `chat_balanced` tier to
   judge the candidate evidence against the question.
2. **Lexical patterns** — for fast, deterministic checks
   (`causal_attribution`, `specific_info_type`) the constraint scans
   for anchor phrases and structural cues without any model call.
3. **Optional embedder slot** — `SemanticMatcher` accepts an embedder
   for language-agnostic anchor-phrase matching, but the production
   path (`create_default_constraints(chat=..., chat_balanced=...)`)
   leaves it unset; the embedding API was removed in v0.12.0. The
   slot remains for tests and downstream consumers that want to plug
   in their own vector backend.

The result is a deterministic constraint cascade where each step is
either pure-Python or a single small chat call.

---

## Creating Custom Constraints

### Protocol

Constraints implement the `ConstraintPlugin` protocol:

```python
from fitz_sage.governance.constraints.base import ConstraintPlugin, ConstraintResult

class MyConstraint:
    @property
    def name(self) -> str:
        return "my_constraint"

    def apply(self, query: str, chunks: list) -> ConstraintResult:
        # Your logic here
        if some_problem_detected:
            return ConstraintResult.deny(
                reason="Explanation for user",
                signal="qualified"  # or "disputed", "abstain"
            )
        return ConstraintResult.allow()
```

### Requirements

Constraints must be:
- **Deterministic**: same input → same output
- **Side-effect free**: no mutation, no network calls beyond the
  optional injected chat client
- **Fast**: at most one chat call per constraint; ideally zero
  (lexical / structural detection)

### Example: Recency Constraint

```python
from datetime import datetime, timedelta
from fitz_sage.governance.constraints.base import ConstraintResult

class RecencyConstraint:
    """Warn when sources are outdated."""

    def __init__(self, max_age_days: int = 365):
        self.max_age = timedelta(days=max_age_days)

    @property
    def name(self) -> str:
        return "recency"

    def apply(self, query: str, chunks: list) -> ConstraintResult:
        now = datetime.utcnow()
        outdated = []

        for chunk in chunks:
            created_at = chunk.metadata.get("created_at")
            if created_at:
                age = now - datetime.fromisoformat(created_at)
                if age > self.max_age:
                    outdated.append(chunk.id)

        if len(outdated) == len(chunks):
            return ConstraintResult.deny(
                reason=f"All sources are over {self.max_age.days} days old",
                signal="qualified",
                outdated_chunks=outdated
            )

        return ConstraintResult.allow()
```

### Registration

Place custom constraints in:
```
fitz_sage/governance/constraints/plugins/my_constraint.py
```

They are auto-discovered.

---

## Comparison with Other Systems

| System          | Uncertainty Handling                         |
|-----------------|----------------------------------------------|
| ChatGPT RAG     | None — answers as if always confident        |
| Generic RAG libs| Prompt-based, easy to bypass with paraphrase |
| **fitz-sage**   | Built-in cascade; emits TRUSTWORTHY / DISPUTED / ABSTAIN |

Fitz treats uncertainty as a **feature**, not a failure.

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_sage/governance/constraints/base.py` | Protocol and result types |
| `fitz_sage/governance/constraints/semantic.py` | Semantic matching utilities |
| `fitz_sage/governance/constraints/plugins/` | Built-in constraints |

---

## See Also

- [ENGINES.md](ENGINES.md) - RAG engine architecture
- [CONFIG.md](CONFIG.md) - Configuration reference
- [PLUGINS.md](PLUGINS.md) - Plugin development
