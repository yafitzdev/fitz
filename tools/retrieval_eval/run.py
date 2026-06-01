# tools/retrieval_eval/run.py
"""Run retrieval benchmarks from PyCharm settings or benchmark CLI arguments."""

from __future__ import annotations

import sys

# --- SETTINGS ---
MODES = None  # e.g. ["section"], ["section", "table"], or None for all
SKIP_INGEST = False  # True reuses the existing collection (skips re-indexing)
UPDATE_BASELINE = False  # True stores this run's scores as the new baseline
REPEATS = 3  # retrieval runs per query, metrics averaged — dampens LLM jitter
VERBOSE = True  # list missed critical units per query
# -----------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        from tools.retrieval_eval.benchmark import main

        main()
    else:
        from tools.retrieval_eval.benchmark import MODES as ALL_MODES
        from tools.retrieval_eval.benchmark import run

        raise SystemExit(
            run(
                MODES or list(ALL_MODES),
                skip_ingest=SKIP_INGEST,
                update=UPDATE_BASELINE,
                verbose=VERBOSE,
                repeats=REPEATS,
            )
        )
