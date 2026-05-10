# tools/retrieval_eval/doc_eval.py
"""
Evaluate document retrieval quality against ground truth.

Tests the RETRIEVAL pipeline in isolation:
  Analysis + Detection + Rewriting + Embedding (parallel LLM calls)
  → Retrieval router (vector + BM25 + RRF)
  → Reranking
  → Content reading (to get section titles/pages)

Skips: synthesis, governance, cloud cache, context expansion.

Uses fitz-sage's own engine with config from ~/.fitz/config.yaml
(ollama/qwen3.5 + nomic-embed-text by default).

Usage:
    python -m tools.retrieval_eval.doc_eval
    python -m tools.retrieval_eval.doc_eval --category section-lookup
    python -m tools.retrieval_eval.doc_eval --ids 1,2,3 -v
    python -m tools.retrieval_eval.doc_eval --skip-ingest

Scoring:
  - critical_recall: % of critical sections found in retrieved results
  - relevant_recall: % of all expected sections found
  - keyword_hit:     % of expected keywords present in retrieved content
  - precision:       % of retrieved sections that match any expected section
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)

GROUND_TRUTH = Path(__file__).parent / "doc_ground_truth.json"
CORPUS_DIR = Path(__file__).parent / "test_corpus"
RESULTS_DIR = Path(__file__).parent / "results"
COLLECTION = "doc_eval"


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _section_matches(read_meta: dict, section: dict) -> bool:
    """Check if a ReadResult matches an expected section.

    ReadResult metadata for sections has:
      - section_title: heading text (e.g. "Naive RAG")
      - page_start / page_end: page numbers
    Address metadata has:
      - location: breadcrumb path (e.g. "Overview of RAG > Naive RAG")
    """
    heading = section.get("heading", "")
    heading_lower = heading.lower()

    # Heading match: check section_title and location (breadcrumb)
    # Also compare with spaces stripped to handle "RiskPrioritization" vs "Risk Prioritization"
    section_title = str(read_meta.get("section_title", ""))
    location = str(read_meta.get("location", ""))
    heading_nospace = heading_lower.replace(" ", "")
    heading_match = heading_lower and (
        heading_lower in section_title.lower()
        or heading_lower in location.lower()
        or heading_nospace in section_title.lower().replace(" ", "")
        or heading_nospace in location.lower().replace(" ", "")
    )

    # Page match (within +/-1)
    expected_page = section.get("page")
    prov_page = read_meta.get("page_start")
    page_match = (
        expected_page is not None
        and prov_page is not None
        and abs(int(prov_page) - int(expected_page)) <= 1
    )

    return heading_match or page_match


def _result_matches_document(meta: dict, document: str) -> bool:
    """Check if a ReadResult is from the expected document."""
    file_path = str(meta.get("file_path", ""))
    location = str(meta.get("location", ""))
    doc_lower = document.lower()
    return doc_lower in file_path.lower() or doc_lower in location.lower()


def _check_keywords(contents: list[str], keywords: list[str]) -> float:
    """Check what fraction of keywords appear in any content."""
    if not keywords:
        return 1.0
    combined = " ".join(contents).lower()
    found = sum(1 for kw in keywords if kw.lower() in combined)
    return found / len(keywords)


def score_query(read_results, query_gt: dict) -> dict:
    """Score retrieval results for a single query.

    read_results: list of ReadResult from ContentReader.read()
    """
    document = query_gt["document"]
    critical = query_gt.get("critical_sections", [])
    relevant = query_gt.get("relevant_sections", [])
    all_expected = critical + relevant

    # Build merged metadata + content for each result
    items = []
    for r in read_results:
        meta = dict(r.metadata)
        # Merge address metadata (has location, page_start, etc.)
        meta.update({k: v for k, v in r.address.metadata.items() if k != "context_type"})
        meta["file_path"] = r.file_path
        meta["location"] = r.address.location
        items.append({"meta": meta, "content": r.content or ""})

    # Filter to matching document
    doc_items = [it for it in items if _result_matches_document(it["meta"], document)]

    # Score critical sections
    critical_found = []
    critical_missed = []
    for section in critical:
        found = any(_section_matches(it["meta"], section) for it in doc_items)
        label = section.get("heading", f"page {section.get('page')}")
        if found:
            critical_found.append(label)
        else:
            critical_missed.append(label)

    # Score all sections
    all_found = []
    for section in all_expected:
        found = any(_section_matches(it["meta"], section) for it in doc_items)
        if found:
            all_found.append(section.get("heading", f"page {section.get('page')}"))

    # Keyword check
    all_keywords = []
    for section in critical:
        all_keywords.extend(section.get("keywords", []))
    keyword_hit = _check_keywords([it["content"] for it in doc_items], all_keywords)

    # Precision
    matched = sum(
        1 for it in doc_items if any(_section_matches(it["meta"], s) for s in all_expected)
    )

    critical_recall = len(critical_found) / len(critical) if critical else 1.0
    total_recall = len(all_found) / len(all_expected) if all_expected else 1.0
    precision = matched / len(doc_items) if doc_items else 0.0

    return {
        "critical_recall": round(critical_recall, 2),
        "critical_found": critical_found,
        "critical_missed": critical_missed,
        "total_recall": round(total_recall, 2),
        "keyword_hit": round(keyword_hit, 2),
        "precision": round(precision, 2),
        "doc_result_count": len(doc_items),
        "total_result_count": len(items),
    }


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------


def _clean_collection(collection: str) -> None:
    """Delete stale manifest + parsed cache so point() re-indexes from scratch."""
    import shutil

    from fitz_sage.core.paths import FitzPaths

    col_dir = FitzPaths.workspace() / "collections" / collection
    if col_dir.exists():
        shutil.rmtree(col_dir)
        print(f"Cleaned collection directory: {col_dir}")


def _wait_for_indexing(engine, timeout: float = 1800) -> None:
    """Wait for background indexing to complete."""
    worker = getattr(engine, "_bg_worker", None)
    if not worker:
        return
    thread = getattr(worker, "_thread", None)
    if not thread or not thread.is_alive():
        return

    print("Waiting for background indexing to complete...")
    t0 = time.monotonic()
    thread.join(timeout=timeout)
    elapsed = time.monotonic() - t0

    if thread.is_alive():
        print(f"  WARNING: Indexing still running after {timeout:.0f}s timeout")
    else:
        print(f"  Indexing complete in {elapsed:.1f}s")


def ingest_corpus(engine, corpus_dir: Path, collection: str) -> bool:
    """Ingest test corpus into a collection. Returns True on success."""
    doc_files = [
        f
        for f in corpus_dir.iterdir()
        if f.suffix.lower() in (".pdf", ".docx", ".pptx", ".md", ".txt", ".csv", ".xlsx")
    ]
    if not doc_files:
        print(f"No documents found in {corpus_dir}")
        print("Add test documents to tools/retrieval_eval/test_corpus/")
        return False

    _clean_collection(collection)

    # IMPORTANT: load() switches the engine's stores to the target collection
    # database BEFORE point() starts writing data. Without this, stores write
    # to the default collection from config.yaml.
    engine.load(collection)

    print(f"Ingesting {len(doc_files)} documents into collection '{collection}'...")
    for f in doc_files:
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")

    t0 = time.monotonic()
    engine.point(source=corpus_dir, collection=collection)
    elapsed = time.monotonic() - t0
    print(f"Manifest built in {elapsed:.1f}s")

    _wait_for_indexing(engine)
    return True


# ---------------------------------------------------------------------------
# Isolated retrieval — runs steps 1-4 of the engine, skips synthesis
# ---------------------------------------------------------------------------


def run_retrieval(engine, query_text: str, top_k: int = 15):
    """Run the retrieval pipeline in isolation and return ReadResults.

    Executes:
      1. Analysis + Detection + Rewriting
      2. Retrieval router (BM25 + keyword strategies)
      3. Reranking
      4. Content reading

    Skips: synthesis, governance, cloud cache, agentic search.
    """
    # Disable agentic strategy — it returns whole-file results that pollute
    # section search. For eval, data is fully indexed so we want pure section search.
    saved_agentic = getattr(engine._retrieval_router, "_agentic_strategy", None)
    engine._retrieval_router._agentic_strategy = None

    sanitized = re.sub(r"<[^>]+>", "", query_text).strip()[:500]

    # 1. Rewrite-first pipeline (mirrors engine.answer dispatch)
    retrieval_query = sanitized
    rewrite_result = None

    if engine._query_rewriter:
        try:
            rewrite_result = engine._query_rewriter.rewrite(sanitized)
            if rewrite_result.rewritten_query != sanitized:
                retrieval_query = rewrite_result.rewritten_query
        except Exception:
            pass

    classify_query = retrieval_query
    fast_analysis = engine._fast_analyze(classify_query)
    need_llm_analysis = fast_analysis is None
    need_detection = bool(
        engine._detection_orchestrator and engine._needs_detection(classify_query)
    )

    detection_limit_to = None
    if need_detection:
        flagged = engine._detection_orchestrator.gate_categories(classify_query)
        if flagged is not None and len(flagged) == 0:
            need_detection = False
        else:
            detection_limit_to = flagged

    if need_llm_analysis or need_detection:
        try:
            batch_result = engine._query_batcher.batch_classify(
                classify_query,
                include_analysis=need_llm_analysis,
                include_detection=need_detection,
                detection_limit_to=detection_limit_to,
                include_rewriting=False,
            )
            analysis = batch_result.analysis if batch_result.analysis else fast_analysis
            detection = (
                engine._build_detection_summary(batch_result.detection_results, classify_query)
                if need_detection and batch_result.detection_results is not None
                else None
            )
        except Exception:
            analysis = fast_analysis
            detection = None
    else:
        analysis = fast_analysis
        detection = None

    # 2. Build retrieval profile and retrieve addresses
    from fitz_sage.engines.fitz_krag.retrieval_profile import build_retrieval_profile

    profile = build_retrieval_profile(analysis, detection, engine._config)

    if engine._hop_controller:
        read_results = engine._hop_controller.execute(retrieval_query, profile)
        addresses = [r.address for r in read_results] if read_results else []
    else:
        addresses = engine._retrieval_router.retrieve(
            retrieval_query,
            profile,
            rewrite_result=rewrite_result,
        )

    if not addresses:
        engine._retrieval_router._agentic_strategy = saved_agentic
        return []

    # 3. Rerank
    if engine._address_reranker and not engine._hop_controller:
        addresses = engine._address_reranker.rerank(retrieval_query, addresses)

    # 4. Read content — restore disabled components
    engine._retrieval_router._agentic_strategy = saved_agentic
    if engine._hop_controller:
        return read_results or []
    return engine._reader.read(addresses, min(len(addresses), top_k))


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------


def run_eval(
    corpus_dir: str | Path = CORPUS_DIR,
    collection: str = COLLECTION,
    category: str | None = None,
    ids: str | None = None,
    top_k: int = 15,
    verbose: bool = True,
    skip_ingest: bool = False,
):
    """Run full document retrieval evaluation."""
    from fitz_sage.runtime import create_engine

    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        print(f"Corpus directory not found: {corpus_dir}")
        return

    queries = json.loads(GROUND_TRUTH.read_text())
    if not queries or (len(queries) == 1 and "EXAMPLE" in queries[0].get("query", "")):
        print("Ground truth contains only the example query.")
        print("Add test documents to test_corpus/ and update doc_ground_truth.json")
        return

    if category:
        queries = [q for q in queries if q.get("category") == category]
    if ids:
        id_set = set(int(i) for i in ids.split(","))
        queries = [q for q in queries if q["id"] in id_set]

    if not queries:
        print("No queries match filters.")
        return

    # Create engine (auto-loads ~/.fitz/config.yaml)
    engine = create_engine("fitz_krag")
    if not skip_ingest:
        if not ingest_corpus(engine, corpus_dir, collection):
            return
    else:
        print(f"Skipping ingestion, loading collection '{collection}'...")
        engine.load(collection)

    print(f"\nRunning {len(queries)} document retrieval evaluations...\n")

    results = []
    for q in queries:
        t0 = time.monotonic()
        try:
            read_results = run_retrieval(engine, q["query"], top_k)
            elapsed = time.monotonic() - t0

            # Debug: dump first result metadata for first query
            if q["id"] == queries[0]["id"] and read_results and verbose:
                print(f"  DEBUG: {len(read_results)} read results")
                for i, r in enumerate(read_results[:3]):
                    meta = dict(r.metadata)
                    meta.update(r.address.metadata)
                    title = meta.get("section_title", meta.get("name", "?"))
                    page = meta.get("page_start", "?")
                    kind = r.address.kind.value
                    fp = r.file_path
                    print(f"    [{i}] kind={kind} file={fp} title={title} page={page}")

            scores = score_query(read_results, q)
            scores["id"] = q["id"]
            scores["query"] = q["query"][:60]
            scores["category"] = q.get("category", "")
            scores["time"] = round(elapsed, 1)

            is_pass = scores["critical_recall"] == 1.0
            status = "PASS" if is_pass else "MISS"
            label = (
                f"crit={scores['critical_recall']:.0%} "
                f"total={scores['total_recall']:.0%} "
                f"kw={scores['keyword_hit']:.0%}"
            )

            print(f"  [{q['id']:2d}] {status} {label} ({elapsed:.1f}s) {q['query'][:55]}")

            if not is_pass and verbose:
                for missed in scores["critical_missed"]:
                    print(f"       MISSED: {missed}")

            results.append(scores)
        except Exception as e:
            import traceback

            elapsed = time.monotonic() - t0
            print(f"  [{q['id']:2d}] FAILED ({elapsed:.1f}s): {e}")
            if verbose:
                traceback.print_exc()
            results.append({"id": q["id"], "error": str(e)})

    # Summary
    scored = [r for r in results if "error" not in r]
    if not scored:
        print("\nAll queries failed.")
        return

    avg_crit = sum(r["critical_recall"] for r in scored) / len(scored)
    avg_total = sum(r["total_recall"] for r in scored) / len(scored)
    avg_kw = sum(r["keyword_hit"] for r in scored) / len(scored)
    avg_prec = sum(r["precision"] for r in scored) / len(scored)
    perfect = sum(1 for r in scored if r["critical_recall"] == 1.0)
    avg_time = sum(r["time"] for r in scored) / len(scored)
    total_time = sum(r["time"] for r in scored)

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(scored)} queries, {total_time:.0f}s total)")
    print(f"{'=' * 60}")
    print(f"Critical recall:  {avg_crit:.0%} avg ({perfect}/{len(scored)} perfect)")
    print(f"Total recall:     {avg_total:.0%} avg")
    print(f"Keyword hit:      {avg_kw:.0%} avg")
    print(f"Precision:        {avg_prec:.0%} avg")
    print(f"Avg time:         {avg_time:.1f}s per query")

    # By category
    categories = sorted(set(r.get("category", "uncategorized") for r in scored))
    if len(categories) > 1:
        print("\nBy category:")
        for cat in categories:
            cat_results = [r for r in scored if r.get("category") == cat]
            if cat_results:
                cat_crit = sum(r["critical_recall"] for r in cat_results) / len(cat_results)
                cat_perfect = sum(1 for r in cat_results if r["critical_recall"] == 1.0)
                cat_time = sum(r["time"] for r in cat_results) / len(cat_results)
                print(
                    f"  {cat:25s} crit={cat_crit:.0%} "
                    f"({cat_perfect}/{len(cat_results)} perfect) "
                    f"avg={cat_time:.1f}s"
                )

    # Most-missed sections
    all_missed = []
    for r in scored:
        all_missed.extend(r.get("critical_missed", []))
    if all_missed:
        from collections import Counter

        missed_counts = Counter(all_missed).most_common(10)
        print("\nMost-missed critical sections:")
        for section, count in missed_counts:
            print(f"  {count}x {section}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"doc_eval_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document retrieval eval")
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--category", default=None)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    run_eval(
        corpus_dir=args.corpus_dir,
        collection=args.collection,
        category=args.category,
        ids=args.ids,
        top_k=args.top_k,
        verbose=args.verbose,
        skip_ingest=args.skip_ingest,
    )
