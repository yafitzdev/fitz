"""
End-to-end smoke test against tests/e2e_krag/fixtures_rag.

Single mode — fitz-sage has one retrieval mode (BM25 + KRAG typed-unit
routing + LLM rerank). Runs five sanity queries spanning the fixtures
and checks both answer content and the governance mode (TRUSTWORTHY /
DISPUTED / ABSTAIN).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from pathlib import Path

# Force project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from fitz_sage.core import Query  # noqa: E402
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig  # noqa: E402
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine  # noqa: E402

CORPUS_DIR = ROOT / "tests" / "e2e_krag" / "fixtures_rag"


# Five sanity queries spanning the fixtures.
# Each one has a known expected answer in the corpus.
QUERIES = [
    {
        "q": "What is the price and range of the Model X100?",
        "expect_substrings": ["45,000", "300"],
        "expect_mode": "trustworthy",
        "label": "factual / single doc",
    },
    {
        "q": "Who is the CEO of TechCorp Industries and what is their background?",
        "expect_substrings": ["Sarah Chen", "MIT"],
        "expect_mode": "trustworthy",
        "label": "factual / entity",
    },
    {
        "q": "How many employees does TechCorp have?",
        "expect_substrings": ["5,200", "4,800"],  # conflict between Finance & HR
        "expect_mode": "disputed",
        "label": "conflict / two-source contradiction",
    },
    {
        "q": "What is the Authentication Service's preferred token rotation interval?",
        "expect_substrings": [],  # not in corpus
        "expect_mode": "abstain",
        "label": "abstain / not in corpus",
    },
    {
        "q": "What was the population of Mars in 2023?",
        "expect_substrings": [],  # totally off-topic
        "expect_mode": "abstain",
        "label": "abstain / off-topic",
    },
]


LM_STUDIO_URL = "http://localhost:1234/v1"


def _build_config(collection: str) -> FitzKragConfig:
    """Build a minimal config pointing at LM Studio (port 1234)."""
    return FitzKragConfig(
        collection=collection,
        # Chat: qwen3.6-27b loaded as 'smoke-chat' identifier in LM Studio.
        chat_fast="endpoint/smoke-chat",
        chat_balanced="endpoint/smoke-chat",
        chat_smart="endpoint/smoke-chat",
        chat_base_url=LM_STUDIO_URL,
        # Default rerank: llm — wires LLMReranker.
        rerank="llm",
        # Disable optional features to keep the smoke fast.
        enable_query_rewriting=False,
        enable_detection=False,
        enable_enrichment=False,
        enable_hierarchy=False,
        strict_grounding=False,
        enable_guardrails=False,
        top_addresses=20,
        top_read=8,
    )


def _check_mode(actual: str, expected: str) -> str:
    return "OK" if actual == expected else "MISS"


def _check_substrings(text: str, expected: list[str]) -> tuple[int, int]:
    text_lc = text.lower()
    hits = sum(1 for s in expected if s.lower() in text_lc)
    return hits, len(expected)


def run_pass() -> dict:
    """Ingest + query against the fixtures. Return summary dict."""
    collection = f"smoke_{uuid.uuid4().hex[:6]}"
    print(f"\n{'='*70}")
    print(f"  PASS: collection={collection}")
    print(f"{'='*70}\n")

    cfg = _build_config(collection)
    t_init0 = time.perf_counter()
    engine = FitzKragEngine(cfg)
    t_init = time.perf_counter() - t_init0
    print(f"  Engine init: {t_init:.1f}s")

    t_ing0 = time.perf_counter()
    manifest = engine.point(CORPUS_DIR, collection, start_worker=False, progress=print)
    t_ing = time.perf_counter() - t_ing0
    print(f"  Ingest: {t_ing:.1f}s, {len(manifest.entries())} files")

    results = []
    for spec in QUERIES:
        t0 = time.perf_counter()
        try:
            answer = engine.answer(Query(text=spec["q"]))
            t_q = time.perf_counter() - t0
            text = answer.text or ""
            mode = answer.mode.value if answer.mode else "unknown"
            hits, total = _check_substrings(text, spec["expect_substrings"])
            mode_ok = _check_mode(mode, spec["expect_mode"])
            print(f"\n  Q: {spec['q']}")
            print(f"     label: {spec['label']}")
            print(f"     mode: {mode}  (expected {spec['expect_mode']})  [{mode_ok}]")
            if total:
                print(f"     substrings: {hits}/{total}")
            print(f"     answer: {text[:240]}{'...' if len(text)>240 else ''}")
            print(f"     time: {t_q:.1f}s")
            results.append(
                {
                    "q": spec["q"],
                    "label": spec["label"],
                    "expected_mode": spec["expect_mode"],
                    "actual_mode": mode,
                    "mode_match": mode == spec["expect_mode"],
                    "substring_hits": hits,
                    "substring_total": total,
                    "time_s": round(t_q, 2),
                    "answer_preview": text[:500],
                }
            )
        except Exception as e:
            print(f"\n  Q: {spec['q']}")
            print(f"     ERROR: {e}")
            traceback.print_exc()
            results.append(
                {
                    "q": spec["q"],
                    "label": spec["label"],
                    "expected_mode": spec["expect_mode"],
                    "error": str(e),
                }
            )

    return {
        "collection": collection,
        "init_s": round(t_init, 2),
        "ingest_s": round(t_ing, 2),
        "files": len(manifest.entries()),
        "queries": results,
    }


def main():
    import httpx

    try:
        r = httpx.get(f"{LM_STUDIO_URL}/models", timeout=2.0)
        models = [m.get("id") for m in r.json().get("data", [])]
        print(f"LM Studio at {LM_STUDIO_URL} — models: {models}")
    except Exception as e:
        print(f"LM Studio not reachable at {LM_STUDIO_URL}: {e}")
        sys.exit(1)

    try:
        result = run_pass()
    except Exception as e:
        print(f"\n!! Smoke test FAILED at top level: {e}")
        traceback.print_exc()
        result = {"error": str(e), "traceback": traceback.format_exc()}

    out_path = ROOT / ".smoke_test" / "results.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n\nResults written to {out_path}")

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    if "error" in result:
        print(f"\n  FAILED — {result['error']}")
        sys.exit(1)
    qs = result["queries"]
    mode_matches = sum(1 for q in qs if q.get("mode_match"))
    sub_hit_total = sum(q.get("substring_hits", 0) for q in qs)
    sub_total = sum(q.get("substring_total", 0) for q in qs)
    avg_q_time = sum(q.get("time_s", 0) for q in qs) / max(1, len(qs))
    print(f"\n    ingest:      {result['ingest_s']:.1f}s ({result['files']} files)")
    print(f"    mode-match:  {mode_matches}/{len(qs)} queries land on the expected mode")
    print(f"    substrings:  {sub_hit_total}/{sub_total} expected substrings present")
    print(f"    avg query:   {avg_q_time:.1f}s")


if __name__ == "__main__":
    main()
