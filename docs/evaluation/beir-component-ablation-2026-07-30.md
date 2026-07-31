# BEIR Component Ablation (2026-07-30)

This is the current external retrieval-component measurement for Fitz-Sage.
The run used commit `2893be4f35cacb67c8ca8627b20f08cf1dfd9817` from a
clean worktree.

## Method

- 66,454 source documents from NFCorpus, FiQA, and SciFact
- all 1,271 judged test queries
- source-only Fitz-Sage indexes with optional document enrichment disabled
- the same index, query order, candidate budgets, compiler, and exact Pyrrho
  runtime for every variant
- isolated Python processes with no model state shared between variants
- query records paired by dataset and query ID
- 2,000-sample paired percentile bootstrap intervals

The four benchmark-only variants were:

| Variant | Managed Qwen query terms | INT8 cross-encoder |
|---|---:|---:|
| `literal` | off | off |
| `expansion` | on | off |
| `reranker` | off | on |
| `full` | on | on |

Disabling the cross-encoder did not bypass the rest of retrieval. A stable
top-k selector preserved its output budget, so evidence reading, compilation,
and Pyrrho received structurally comparable inputs.

The paired measurement-integrity gate passed. Every variant used the same Git
state, dataset hashes, collection identities, cutoff set, baseline scores,
query IDs, and query order.

## Quality

Final-candidate nDCG@10:

| Dataset | Plain BM25 | `literal` | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|---:|
| NFCorpus | 0.3062 | 0.3179 | 0.3265 | 0.3392 | 0.3377 |
| FiQA | 0.2377 | 0.2490 | 0.2459 | 0.3170 | 0.3188 |
| SciFact | 0.6634 | 0.6457 | 0.6438 | 0.6556 | 0.6529 |
| Macro mean | - | 0.4042 | 0.4054 | 0.4372 | 0.4365 |

Delivered-evidence nDCG@10:

| Dataset | Plain BM25 | `literal` | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|---:|
| NFCorpus | 0.3062 | 0.3172 | 0.3257 | 0.3382 | 0.3367 |
| FiQA | 0.2377 | 0.2480 | 0.2447 | 0.3160 | 0.3179 |
| SciFact | 0.6634 | 0.6053 | 0.6045 | 0.6191 | 0.6170 |
| Macro mean | - | 0.3902 | 0.3916 | 0.4244 | 0.4239 |

Recall@50 before reranking:

| Dataset | Plain BM25 | `literal` | Qwen-enabled |
|---|---:|---:|---:|
| NFCorpus | 0.2101 | 0.2164 | 0.2231 |
| FiQA | 0.4459 | 0.4746 | 0.4766 |
| SciFact | 0.8704 | 0.8952 | 0.8909 |

The literal Fitz-Sage recall path exceeded the whole-document BM25 Recall@50
on all three datasets before Qwen terms were added. This isolates the value of
typed indexing, deterministic query planning, and retrieval fusion from the
managed query model.

### Reranker effect

With Qwen disabled, the cross-encoder changed final nDCG@10 by:

| Dataset | Mean delta | Paired 95% interval |
|---|---:|---:|
| NFCorpus | +0.0213 | [+0.0088, +0.0334] |
| FiQA | +0.0680 | [+0.0495, +0.0863] |
| SciFact | +0.0099 | [-0.0287, +0.0471] |

The reranker has a clear positive effect on NFCorpus and FiQA. The SciFact
effect is positive on average but inconclusive. Across datasets, reranking
added 2.10 seconds to mean query latency and improved macro final nDCG@10 by
0.0330.

### Qwen expansion effect

With the reranker disabled, Qwen changed recall nDCG@10 by:

| Dataset | Mean delta | Paired 95% interval |
|---|---:|---:|
| NFCorpus | +0.0087 | [+0.0031, +0.0152] |
| FiQA | -0.0027 | [-0.0087, +0.0034] |
| SciFact | -0.0005 | [-0.0109, +0.0101] |

Qwen produced a measurable recall-ordering gain on NFCorpus and no conclusive
gain on FiQA or SciFact. Its Recall@50 changes were inconclusive on all three
datasets. With reranking already enabled, Qwen changed macro final nDCG@10 by
-0.0008 and delivered nDCG@10 by -0.0005 while adding 1.98 seconds.

This broad run alone did not justify deleting semantic expansion because it
was not a targeted test of ordinary synonym and paraphrase mismatch. It did
show that broad BEIR quality could not justify paying the Qwen cost on every
query.

The required follow-up is now recorded in the
[BEIR Semantic Holdout](beir-semantic-holdout-2026-07-30.md). That frozen
ArguAna/Quora run found that the current Qwen path did not earn its cost on
those tasks. It does not decide whether query expansion belongs in the default
company-document retrieval path, where BM25 still has no semantic matching of
its own.

## Latency

Mean query latency:

| Dataset | `literal` | `expansion` | `reranker` | `full` |
|---|---:|---:|---:|---:|
| NFCorpus | 1.83s | 3.90s | 4.12s | 5.98s |
| FiQA | 2.33s | 4.23s | 3.89s | 5.58s |
| SciFact | 2.06s | 3.95s | 4.53s | 6.90s |
| Macro mean | 2.07s | 4.03s | 4.18s | 6.15s |

Across datasets, Qwen added about 1.95 to 1.98 seconds and reranking added
about 2.10 to 2.13 seconds. The exact Pyrrho evidence decision consumed about
1.4 to 1.7 seconds of the literal path. Lexical recall itself averaged roughly
0.16 to 0.38 seconds.

The four-variant run required about 6.3 hours of wall time on the benchmark
machine. Query checkpoints allowed the interrupted final variant to resume
without recomputing completed records. The `full` variant therefore spans two
process lifetimes and includes one additional cold-inference boundary across
1,271 records; this has negligible aggregate impact but is part of the timing
record.

## Boundaries

FiQA contains 38 upstream records with empty title and text fields, including
one judged-relevant record. Fitz-Sage reported all 38 as unsearchable instead
of indexing fabricated content. The individual source reports therefore have
an operational ingestion warning; the paired measurement-integrity gate still
passed.

On SciFact, 13 queries had a judged-relevant final candidate but no judged
relevant compiled evidence. Every one contained a literal structured
identifier, including examples such as `anti-interleukin-2`, `SHP-2`,
`CK-666`, and `FOXO3`. Fitz-Sage intentionally requires those identifiers to
occur literally in compiled evidence. It does not guess that separator,
abbreviation, or naming variants are equivalent. Corpus cleanup or an explicit
user-owned preprocessing mapping owns that equivalence. A public vocabulary
hook is deferred and is not part of the current API.

Compiled and delivered rankings were identical on all three datasets. The
exact Pyrrho output was recorded unchanged and was not treated as a relevance
label.

## Reproduce

```bash
python -m benchmarks.fitz_bench.beir_ablation \
  --offline \
  --no-resume-queries
```

The detailed JSON and Markdown reports are generated under
`benchmarks/results/`, which is intentionally ignored because query-level
reports are large.
