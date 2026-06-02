# docs/evaluation/

Benchmark results and analysis for fitz-sage retrieval and governance.

| File | Contents |
|---|---|
| `beir-results.md` | BEIR benchmark scores for retrieval quality |
| `fitz-gov-5.0-results.md` | Governance pipeline v5.0 evaluation results |

## Benchmarks tracked

- **Retrieval** — BEIR (NDCG@10 across 14 datasets), hybrid BM25+dense vs dense-only
- **Governance** — precision/recall on constraint enforcement, cascade calibration
- **Detection classifier** — temporal and comparison query classification (recall-oriented)

For the tools used to produce these results, see `tools/detection/` and `tools/governance/`.
