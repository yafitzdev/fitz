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
from .metrics import ndcg_at_k, recall_at_k

HERE = Path(__file__).resolve().parent
DATASETS_DIR = HERE / "datasets"
RESULTS_DIR = HERE / "results"
BASELINE_PATH = HERE / "baseline.json"

K_VALUES = (5, 10, 20)
PRIMARY_K = 10
DEFAULT_TOLERANCE = 0.03
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
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(engine, dataset: Dataset, verbose: bool) -> list[dict]:
    """Run every query through ``engine.retrieve()`` and rank the ground truth."""
    from fitz_sage.core import Query

    records: list[dict] = []
    for q in dataset.queries:
        t0 = time.monotonic()
        try:
            results = engine.retrieve(Query(text=q.query))
        except Exception as e:  # noqa: BLE001 — one bad query must not abort the run
            print(f"  [{q.id}] ERROR: {e}")
            records.append({"id": q.id, "query": q.query, "error": str(e)})
            continue

        units = rank_units(dataset.mode, results, q.relevant)
        record = {
            "id": q.id,
            "query": q.query,
            "vocab_mismatch": q.vocab_mismatch,
            "units": units,
            "n_results": len(results),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }
        records.append(record)

        print(
            f"  [{q.id}] recall@{PRIMARY_K}={recall_at_k(units, PRIMARY_K):.0%} "
            f"ndcg@{PRIMARY_K}={ndcg_at_k(units, PRIMARY_K):.2f} "
            f"({record['elapsed_s']}s) {q.query[:48]}"
        )
        if verbose:
            for unit, gt in zip(units, q.relevant):
                if unit.grade >= 2 and (unit.rank is None or unit.rank > PRIMARY_K):
                    print(f"        missed critical: {_unit_label(gt)}")
    return records


def _unit_label(unit: dict) -> str:
    return unit.get("path") or unit.get("heading") or unit.get("value") or "?"


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def aggregate(records: list[dict]) -> dict:
    """Mean recall@k / nDCG@k across the scored (non-error) records."""
    scored = [r for r in records if "units" in r]
    if not scored:
        return {}
    out: dict[str, float] = {}
    for k in K_VALUES:
        out[f"recall@{k}"] = _mean(recall_at_k(r["units"], k) for r in scored)
        out[f"ndcg@{k}"] = _mean(ndcg_at_k(r["units"], k) for r in scored)
    # Critical recall is averaged only over queries that have critical units.
    crit = [r for r in scored if any(u.grade >= 2 for u in r["units"])]
    for k in K_VALUES:
        if crit:
            out[f"recall@{k}_critical"] = _mean(
                recall_at_k(r["units"], k, min_grade=2) for r in crit
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
    dataset: Dataset, records: list[dict], baseline: dict, *, indexing_ok: bool = True
) -> str:
    """Print the per-mode report; return "ok", "regression", "below-threshold" or "error"."""
    scored = [r for r in records if "units" in r]
    errors = [r for r in records if "error" in r]
    n_vocab = sum(1 for q in dataset.queries if q.vocab_mismatch)

    print()
    print("=" * 64)
    print(f"RETRIEVAL BENCHMARK — {dataset.mode}")
    print("=" * 64)
    print(f"queries        : {len(dataset.queries)}  ({n_vocab} vocab-mismatch)")
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
                "n_results": r["n_results"],
                "elapsed_s": r["elapsed_s"],
                "units": [{"grade": u.grade, "rank": u.rank} for u in r["units"]],
            }
        )
    path = RESULTS_DIR / f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def update_baseline(baseline: dict, mode: str, overall: dict) -> None:
    """Store this run's scores as the new baseline for ``mode``."""
    modes = baseline.setdefault("modes", {})
    mode_bl = modes.setdefault(
        mode, {"thresholds": {f"recall@{PRIMARY_K}": None, f"ndcg@{PRIMARY_K}": None}}
    )
    mode_bl["scores"] = overall
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

        records = evaluate(engine, dataset, verbose)
        status = report(dataset, records, baseline, indexing_ok=indexing_ok)
        out_path = save_results(mode, records)
        print(f"results        {out_path.relative_to(HERE.parents[1])}")

        if status == "error":
            exit_code = 1
        else:
            if update:
                overall = aggregate(records)
                if overall:
                    update_baseline(baseline, mode, overall)
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
    parser.add_argument("-v", "--verbose", action="store_true", help="list missed critical units")
    args = parser.parse_args()
    sys.exit(
        run(
            args.mode or list(MODES),
            skip_ingest=args.skip_ingest,
            update=args.update_baseline,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
