# tools/retrieval_eval/_spike_run.py
"""v-0-13-08 spike A/B: section retrieval with LLM summaries vs distilbart summaries.

Arm A = reval_section       (LLM section summaries)
Arm B = reval_section_slm   (distilbart-cnn-12-6 summaries, same corpus/index)

Both collections are already on disk; this only runs the eval phase. The
reranker reads Address.summary, so the summary column is the only variable.
"""
from fitz_sage.runtime import create_engine

from tools.retrieval_eval.benchmark import DATASETS_DIR, aggregate, evaluate
from tools.retrieval_eval.dataset import Dataset, load_dataset

REPEATS = 3
ARMS = [("LLM summaries", "reval_section"), ("SLM summaries", "reval_section_slm")]

base = load_dataset(DATASETS_DIR / "section.json")
engine = create_engine("fitz_krag")

results: dict[str, dict] = {}
per_query: dict[str, dict] = {}
for arm, collection in ARMS:
    print(f"\n{'=' * 64}\nARM: {arm}  (collection={collection})\n{'=' * 64}", flush=True)
    ds = Dataset(mode=base.mode, corpus=base.corpus, collection=collection, queries=base.queries)
    engine.load(collection)
    records = evaluate(engine, ds, False, REPEATS)
    results[arm] = aggregate(records)
    per_query[arm] = {r["id"]: r["metrics"]["recall@10"] for r in records if "metrics" in r}

print(f"\n{'=' * 64}\nA/B RESULT — section retrieval (repeats={REPEATS})\n{'=' * 64}")
for arm, _ in ARMS:
    o = results[arm]
    print(
        f"{arm:16} recall@5={o['recall@5']:.4f}  recall@10={o['recall@10']:.4f}  "
        f"recall@20={o['recall@20']:.4f}  ndcg@10={o['ndcg@10']:.4f}"
    )
a, b = results["LLM summaries"], results["SLM summaries"]
print(
    f"{'delta SLM-LLM':16} recall@10={b['recall@10'] - a['recall@10']:+.4f}  "
    f"ndcg@10={b['ndcg@10'] - a['ndcg@10']:+.4f}"
)

print("\nper-query recall@10 (queries where the two arms differ):")
llm_q, slm_q = per_query["LLM summaries"], per_query["SLM summaries"]
for qid in sorted(llm_q):
    lv, sv = llm_q[qid], slm_q.get(qid, 0.0)
    if abs(sv - lv) > 0.01:
        print(f"  {qid:16} LLM={lv:.2f}  SLM={sv:.2f}  ({sv - lv:+.2f})")
