# tools/retrieval_eval/benchmark.py
"""Retrieval benchmark — confirm fitz-sage retrieves competently, not maximally.

Scores ``engine.retrieve()`` directly (the stable public retrieval API) against
curated ground truth, with deterministic recall@k / nDCG@k. The numbers are a
floor and a regression alarm — "are we still good, did this change regress" —
never a leaderboard to climb.

Usage:
    python -m tools.retrieval_eval.benchmark                  # every mode
    python -m tools.retrieval_eval.benchmark --mode section
    python -m tools.retrieval_eval.benchmark --mode section --skip-ingest
    python -m tools.retrieval_eval.benchmark --repeats 5      # average 5 runs/query
    python -m tools.retrieval_eval.benchmark --update-baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from .dataset import MODES, Dataset, load_dataset
from .matchers import rank_units
from .metrics import Unit, ndcg_at_k, recall_at_k

HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE / "datasets"
RESULTS_DIR = HERE / "results"
BASELINE_PATH = HERE / "baseline.json"

K_VALUES = (5, 10, 20)
PRIMARY_K = 10
DEFAULT_TOLERANCE = 0.03
DEFAULT_REPEATS = 3
INDEXING_TIMEOUT = 3600.0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def _clean_collection(collection: str) -> None:
    """Drop a collection's manifest, parse cache, and SQLite store for a clean run."""
    from fitz_sage.core.paths import FitzPaths

    workspace = FitzPaths.workspace()
    col_dir = workspace / "collections" / collection
    if col_dir.exists():
        shutil.rmtree(col_dir)
    sqlite_dir = workspace / "sqlite"
    if sqlite_dir.exists():
        for db in sqlite_dir.glob(f"fitz_{collection}.db*"):
            db.unlink()


def _wait_for_indexing(engine, timeout: float = INDEXING_TIMEOUT) -> bool:
    """Block until background indexing finishes. Returns False if it timed out."""
    worker = getattr(engine, "_bg_worker", None)
    thread = getattr(worker, "_thread", None) if worker else None
    if not thread or not thread.is_alive():
        return True
    print("  indexing in background...")
    t0 = time.monotonic()
    thread.join(timeout=timeout)
    if thread.is_alive():
        print(f"  WARNING: indexing still running after {timeout:.0f}s — index incomplete")
        return False
    print(f"  indexed in {time.monotonic() - t0:.1f}s")
    return True


def ingest(engine, dataset: Dataset) -> bool:
    """Re-index a dataset's corpus into a clean collection.

    Returns False if background indexing did not finish within the timeout.
    """
    print(f"ingesting {dataset.corpus} -> collection '{dataset.collection}'")
    _clean_collection(dataset.collection)
    engine.load(dataset.collection)
    t0 = time.monotonic()
    engine.point(source=dataset.corpus, collection=dataset.collection)
    print(f"  manifest built in {time.monotonic() - t0:.1f}s")
    return _wait_for_indexing(engine)


# ---------------------------------------------------------------------------
# Feature production check
# ---------------------------------------------------------------------------

# Enrichment features the progressive worker should *reliably* produce, per
# mode — used to flag a silent-absence regression. Code mode expects nothing:
# bare code symbols rarely carry NER-style named entities, so an empty entity
# graph there is content-driven, not a regression. Tables are not enriched.
_EXPECTED_FEATURES = {
    "code": (),
    "section": ("entity_graph", "hierarchy_l1", "hierarchy_l2"),
    "table": (),
}


def check_features(dataset: Dataset) -> dict:
    """Report whether the worker actually produced its corpus enrichment features.

    entity-graph and L1/L2 hierarchy are populated by the background worker
    during ``point()``. If the worker stops producing them, their retrieval
    consumers silently no-op on empty stores — a regression invisible to
    recall/nDCG alone. This makes that absence loud and explicit.
    """
    from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore
    from fitz_sage.retrieval.entity_graph.store import EntityGraphStore
    from fitz_sage.storage.sqlite import SqliteConnectionManager

    col = dataset.collection
    counts: dict[str, object] = {}

    def _safe(label: str, fn) -> None:
        try:
            counts[label] = fn()
        except Exception as e:  # noqa: BLE001 — a store error must not abort the run
            counts[label] = f"error: {e}"

    _safe("entity_graph", lambda: EntityGraphStore(collection=col).stats().get("entities", 0))
    if dataset.mode == "section":
        store = SectionStore(SqliteConnectionManager.get_instance(), col)
        _safe("hierarchy_l1", lambda: len(store.get_hierarchy_summaries()))
        _safe("hierarchy_l2", lambda: len(store.get_corpus_summaries()))

    expected = _EXPECTED_FEATURES.get(dataset.mode, ())
    print()
    print(f"feature production — {dataset.mode}")
    for label, value in counts.items():
        if label not in expected:
            note = "   (informational — not a regression signal for this mode)"
        elif isinstance(value, int) and value == 0:
            note = "   !! EMPTY — enrichment produced nothing (feature regressed)"
        else:
            note = ""
        print(f"  {label:<14} {value}{note}")
    return counts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(engine, dataset: Dataset, verbose: bool, repeats: int) -> list[dict]:
    """Run every query through ``engine.retrieve()`` ``repeats`` times and score it.

    Retrieval runs an LLM somewhere on every path (query prep, and the whole
    ranking for code mode), so one run is one sample of a distribution. Each
    query is retrieved ``repeats`` times and its metrics averaged, so the
    reported number is a stable mean rather than a single jittery draw.
    """
    from fitz_sage.core import Query

    repeats = max(1, repeats)
    records: list[dict] = []
    for q in dataset.queries:
        t0 = time.monotonic()
        runs: list[list[Unit]] = []
        result_counts: list[int] = []
        error: str | None = None
        for _ in range(repeats):
            try:
                results = engine.retrieve(Query(text=q.query))
            except Exception as e:  # noqa: BLE001 — one bad query must not abort the run
                error = str(e)
                break
            runs.append(rank_units(dataset.mode, results, q.relevant))
            result_counts.append(len(results))

        if error is not None:
            print(f"  [{q.id}] ERROR: {error}")
            records.append({"id": q.id, "query": q.query, "error": error})
            continue

        metrics = _query_metrics(runs)
        record = {
            "id": q.id,
            "query": q.query,
            "vocab_mismatch": q.vocab_mismatch,
            "metrics": metrics,
            "has_critical": any(u.grade >= 2 for u in runs[0]),
            "units": runs[-1],
            "n_runs": len(runs),
            "n_results": round(_mean(result_counts)),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        records.append(record)

        print(
            f"  [{q.id}] recall@{PRIMARY_K}={metrics[f'recall@{PRIMARY_K}']:.0%} "
            f"ndcg@{PRIMARY_K}={metrics[f'ndcg@{PRIMARY_K}']:.2f} "
            f"({record['elapsed_s']}s, {len(runs)}x) {q.query[:48]}"
        )
        if verbose:
            for i, gt in enumerate(q.relevant):
                if gt["grade"] < 2:
                    continue
                ranks = [run[i].rank for run in runs]
                if all(r is None or r > PRIMARY_K for r in ranks):
                    print(f"        missed critical (every run): {_unit_label(gt)}")
    return records


def _unit_label(unit: dict) -> str:
    return unit.get("path") or unit.get("heading") or unit.get("value") or "?"


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _query_metrics(runs: list[list[Unit]]) -> dict[str, float]:
    """Per-query recall@k / nDCG@k, averaged over the repeated retrieval runs.

    Averaging happens at the metric level, not the rank level: a unit found at
    rank 2 in one run and missed in the next has no meaningful "mean rank", but
    its recall and nDCG contributions do average. ``recall@k_critical`` is
    computed for every query; ``aggregate`` only folds it in for queries that
    actually have critical units.
    """
    m: dict[str, float] = {}
    for k in K_VALUES:
        m[f"recall@{k}"] = _mean(recall_at_k(u, k) for u in runs)
        m[f"ndcg@{k}"] = _mean(ndcg_at_k(u, k) for u in runs)
        m[f"recall@{k}_critical"] = _mean(recall_at_k(u, k, min_grade=2) for u in runs)
    return m


def aggregate(records: list[dict]) -> dict:
    """Mean recall@k / nDCG@k across the scored records' per-query metrics."""
    scored = [r for r in records if "metrics" in r]
    if not scored:
        return {}
    out: dict[str, float] = {}
    for k in K_VALUES:
        out[f"recall@{k}"] = _mean(r["metrics"][f"recall@{k}"] for r in scored)
        out[f"ndcg@{k}"] = _mean(r["metrics"][f"ndcg@{k}"] for r in scored)
    # Critical recall is averaged only over queries that have critical units.
    crit = [r for r in scored if r["has_critical"]]
    if crit:
        for k in K_VALUES:
            out[f"recall@{k}_critical"] = _mean(
                r["metrics"][f"recall@{k}_critical"] for r in crit
            )
    return out


# ---------------------------------------------------------------------------
# Reporting + baseline
# ---------------------------------------------------------------------------


def load_baseline() -> dict:
    """Read baseline.json, or a fresh skeleton if it does not exist yet."""
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {"primary_k": PRIMARY_K, "regression_tolerance": DEFAULT_TOLERANCE, "modes": {}}


def _fmt_row(label: str, metrics: dict, *, suffix: str = "", with_ndcg: bool = True) -> None:
    parts = [f"{label:<15}"]
    for k in K_VALUES:
        key = f"recall@{k}{suffix}"
        if key in metrics:
            parts.append(f"recall@{k}={metrics[key]:.2f}")
    if with_ndcg and f"ndcg@{PRIMARY_K}" in metrics:
        parts.append(f"nDCG@{PRIMARY_K}={metrics[f'ndcg@{PRIMARY_K}']:.2f}")
    print("  " + "  ".join(parts))


def report(
    dataset: Dataset,
    records: list[dict],
    baseline: dict,
    *,
    indexing_ok: bool = True,
    repeats: int = DEFAULT_REPEATS,
) -> str:
    """Print the per-mode report; return "ok", "regression", "below-threshold" or "error"."""
    scored = [r for r in records if "metrics" in r]
    errors = [r for r in records if "error" in r]
    n_vocab = sum(1 for q in dataset.queries if q.vocab_mismatch)

    print()
    print("=" * 64)
    print(f"RETRIEVAL BENCHMARK — {dataset.mode}")
    print("=" * 64)
    print(f"queries        : {len(dataset.queries)}  ({n_vocab} vocab-mismatch)")
    print(f"runs per query : {repeats}  (metrics averaged)")
    print(f"scored / errors: {len(scored)} / {len(errors)}")
    if not scored:
        print("no queries scored — cannot report")
        return "error"

    overall = aggregate(records)
    vocab = aggregate([r for r in scored if r.get("vocab_mismatch")])

    print()
    _fmt_row("overall", overall)
    _fmt_row("critical", overall, suffix="_critical", with_ndcg=False)
    if vocab:
        _fmt_row("vocab-mismatch", vocab)

    # An unbuilt index makes every query score 0 — that is an infrastructure
    # artifact, not a retrieval measurement. Flag it and refuse to score.
    if not indexing_ok or all(r.get("n_results", 0) == 0 for r in scored):
        print()
        print("!! INDEX NOT READY — the numbers above are an infrastructure artifact,")
        print("!! not a retrieval measurement. Baseline not updated.")
        if not indexing_ok:
            print("!! cause: background indexing did not finish within the timeout.")
        else:
            print("!! cause: every query returned 0 results — the collection index is empty.")
        return "error"

    return _baseline_section(dataset.mode, overall, baseline)


def _baseline_section(mode: str, overall: dict, baseline: dict) -> str:
    mode_bl = baseline.get("modes", {}).get(mode, {})
    scores = mode_bl.get("scores") or {}
    thresholds = mode_bl.get("thresholds") or {}
    tolerance = baseline.get("regression_tolerance", DEFAULT_TOLERANCE)
    rk, nk = f"recall@{PRIMARY_K}", f"ndcg@{PRIMARY_K}"
    status = "ok"

    print()
    if scores:
        d_r = overall[rk] - scores.get(rk, 0.0)
        d_n = overall[nk] - scores.get(nk, 0.0)
        print(
            f"baseline       {rk}={scores.get(rk, 0):.3f}  "
            f"{nk}={scores.get(nk, 0):.3f}   (set {mode_bl.get('updated', '?')})"
        )
        regressed = d_r < -tolerance or d_n < -tolerance
        print(
            f"  delta        {rk}={d_r:+.3f}  {nk}={d_n:+.3f}   "
            f"{'REGRESSION' if regressed else 'ok'}"
        )
        if regressed:
            status = "regression"
    else:
        print("baseline       none — run with --update-baseline to set one")

    floor_r, floor_n = thresholds.get(rk), thresholds.get(nk)
    if floor_r is None and floor_n is None:
        print("threshold      not calibrated — set after a trusted measurement")
    else:
        ok_r = floor_r is None or overall[rk] >= floor_r
        ok_n = floor_n is None or overall[nk] >= floor_n
        passed = ok_r and ok_n
        print(
            f"threshold      {rk}>={floor_r}  {nk}>={floor_n}   "
            f"{'PASS' if passed else 'FAIL — below not-junk floor'}"
        )
        if not passed:
            status = "below-threshold"
    return status


def save_results(mode: str, records: list[dict]) -> Path:
    """Persist per-query results (units serialized) to results/<mode>_<ts>.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = []
    for r in records:
        if "error" in r:
            out.append({"id": r["id"], "error": r["error"]})
            continue
        out.append(
            {
                "id": r["id"],
                "query": r["query"],
                "vocab_mismatch": r["vocab_mismatch"],
                "n_runs": r["n_runs"],
                "n_results": r["n_results"],
                "elapsed_s": r["elapsed_s"],
                "metrics": r["metrics"],
                "units": [{"grade": u.grade, "rank": u.rank} for u in r["units"]],
            }
        )
    path = RESULTS_DIR / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def update_baseline(baseline: dict, mode: str, overall: dict, repeats: int) -> None:
    """Store this run's scores as the new baseline for ``mode``."""
    modes = baseline.setdefault("modes", {})
    mode_bl = modes.setdefault(
        mode, {"thresholds": {f"recall@{PRIMARY_K}": None, f"ndcg@{PRIMARY_K}": None}}
    )
    mode_bl["scores"] = overall
    mode_bl["repeats"] = repeats
    mode_bl["updated"] = time.strftime("%Y-%m-%d")
    baseline["primary_k"] = PRIMARY_K
    baseline.setdefault("regression_tolerance", DEFAULT_TOLERANCE)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    modes: list[str],
    *,
    skip_ingest: bool = False,
    update: bool = False,
    verbose: bool = False,
    repeats: int = DEFAULT_REPEATS,
) -> int:
    """Run the benchmark for the given modes. Returns a process exit code."""
    from fitz_sage.runtime import create_engine

    baseline = load_baseline()
    try:
        engine = create_engine("fitz_krag")
    except Exception as e:  # noqa: BLE001
        print(f"could not create engine: {e}")
        print("the benchmark needs ~/.fitz/config.yaml and a reachable chat model.")
        return 1

    exit_code = 0
    for mode in modes:
        ds_path = DATASETS_DIR / f"{mode}.json"
        if not ds_path.exists():
            print(f"skipping {mode}: no dataset at {ds_path}")
            continue

        dataset = load_dataset(ds_path)
        indexing_ok = True
        if skip_ingest:
            engine.load(dataset.collection)
        else:
            indexing_ok = ingest(engine, dataset)

        check_features(dataset)
        records = evaluate(engine, dataset, verbose, repeats)
        status = report(dataset, records, baseline, indexing_ok=indexing_ok, repeats=repeats)
        out_path = save_results(mode, records)
        print(f"results        {out_path.relative_to(HERE.parents[1])}")

        if status == "error":
            exit_code = 1
        else:
            if update:
                overall = aggregate(records)
                if overall:
                    update_baseline(baseline, mode, overall, repeats)
                    print(f"baseline       updated for {mode}")
            if status in ("regression", "below-threshold"):
                exit_code = 1

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="fitz-sage retrieval benchmark")
    parser.add_argument(
        "--mode", choices=MODES, action="append", help="mode(s) to run (default: all)"
    )
    parser.add_argument("--skip-ingest", action="store_true", help="reuse the existing collection")
    parser.add_argument(
        "--update-baseline", action="store_true", help="store this run as the baseline"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"retrieval runs per query, metrics averaged (default {DEFAULT_REPEATS})",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="list missed critical units")
    args = parser.parse_args()
    sys.exit(
        run(
            args.mode or list(MODES),
            skip_ingest=args.skip_ingest,
            update=args.update_baseline,
            verbose=args.verbose,
            repeats=args.repeats,
        )
    )


if __name__ == "__main__":
    main()
