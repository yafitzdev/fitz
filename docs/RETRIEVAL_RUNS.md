# Retrieval Execution Records

`RetrievalRun` is the versioned audit record for one governed retrieval. It
captures what Fitz actually did, not a second diagnostic execution.

The record is useful for:

- explaining why a query produced a particular `EvidencePack`;
- comparing query planning, candidates, and governance across releases;
- preserving evidence for later governance evaluation;
- debugging production failures without adding hidden data normalization.

## Capture

### CLI

```bash
# Source bodies are redacted by default.
fitz retrieve "Which test failed?" -c reports --trace run.json

# Required when the trace will be used for governance replay.
fitz retrieve "Which test failed?" -c reports \
  --trace run-with-content.json \
  --trace-content
```

`--trace` and normal evidence output come from the same pipeline execution.
Fitz does not retrieve twice.

### SDK

```python
from fitz_sage import trace

run = trace("Which test failed?", collection="reports")
run.write("run.json")  # redacted
run.write("run-with-content.json", include_content=True)
```

The stateful SDK exposes the same operation:

```python
from fitz_sage import fitz

f = fitz(collection="reports")
run = f.trace("Which test failed?")
```

## Inspect

```bash
fitz explain run.json
```

`explain` reads the record and renders a deterministic summary. It does not
load the collection, call a model, or rerun retrieval.

Programmatically:

```python
from fitz_sage import RetrievalRun

run = RetrievalRun.read("run.json")
print(run.explain())
```

## Recorded Stages

A version 1 record contains:

- original, sanitized, and effective retrieval queries;
- query shape and stable planning fields;
- retrieval terms with `literal`, `deterministic`, `query_intelligence`, or
  `semantic` origin;
- retrieval strategy calls and result counts;
- ordered candidate identities and scores at recall, rerank, and final stages;
- the complete ranked evidence after evidence compilation and before cutoff;
- the selected `EvidencePack`;
- every recorded governance-prefix verdict, probability, token count, and stop
  reason that the provider exposed;
- Fitz, engine, collection, component, config-hash, collection-fingerprint, and
  indexing-state metadata.

The collection fingerprint hashes the collection manifest and recorded
indexing status. It is a change detector, not a byte-for-byte snapshot of the
SQLite indexes.

Multi-hop runs remain valid records, but currently include a warning because
the hop controller does not expose every per-hop candidate boundary.

## Redaction

`RetrievalRun.to_dict()`, `to_json()`, and `write()` redact source content by
default. Redacted output removes:

- selected evidence content and excerpts;
- frozen ranked-evidence content and address summaries;
- source-derived item metadata;
- the legacy raw retrieval trace embedded in `EvidencePack.metadata`.

It retains the query, generated terms, source IDs, file paths, structural
locations, scores, hashes, and governance output. A redacted trace is therefore
content-free, but it is not anonymous. Treat query and path metadata according
to your environment's data policy.

`include_content=True` includes the complete compiled evidence and must be
handled like the source documents themselves. Content-bearing records include
SHA-256 digests and character counts. Loading rejects changed frozen evidence.

## Governance Replay

Governance replay asks a Pyrrho provider to evaluate the exact compiled evidence
stored in a content-bearing trace:

```bash
# Use the provider recorded in the trace.
fitz replay run-with-content.json

# Evaluate a different reviewed package.
fitz replay run-with-content.json \
  --governance pyrrho/C:/models/pyrrho-candidate \
  --output replay.json
```

```python
from fitz_sage import replay_governance

result = replay_governance(
    "run-with-content.json",
    governance="pyrrho/C:/models/pyrrho-candidate",
)
print(result.explain())
```

Replay uses the same cutoff implementation, recorded query shape, compiler
metadata, and maximum evidence prefix. It does not rerun query preparation,
BM25, semantic keyword generation, retrieval, reranking, evidence closure, or
compilation. This boundary is deliberate: replay answers "Would this governance
provider judge the same frozen evidence differently?", not "Would the current
system retrieve the same evidence?"

Replay records both the source and current Fitz versions. A version difference
means cutoff implementation changes may also contribute to a changed result.
Governance replay currently supports `fitz_krag` records only.

Redacted traces cannot be replayed. Fitz fails explicitly instead of fetching
current source content, because doing so would no longer be a frozen-evidence
experiment.

## Versioning

The top-level `schema_version` currently uses major version `1`. Readers accept
minor additions within the same major version and ignore unknown fields. They
reject unsupported major versions rather than guessing at changed semantics.

Both `RetrievalRun` and `GovernanceReplay` support `to_dict`, `to_json`,
`write`, `from_dict`, `from_json`, and `read`.
