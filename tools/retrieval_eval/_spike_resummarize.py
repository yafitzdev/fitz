# tools/retrieval_eval/_spike_resummarize.py
"""v-0-13-08 spike: clone reval_section -> reval_section_slm, regenerate the
``summary`` column with distilbart-cnn-12-6 (CPU). FTS indexes title+content,
not summary, so the index stays valid — only the reranker input changes.
"""
import shutil
import sqlite3
import time

from transformers import pipeline

from fitz_sage.core.paths import FitzPaths

sqlite_dir = FitzPaths.workspace() / "sqlite"
src = sqlite_dir / "fitz_reval_section.db"
dst = sqlite_dir / "fitz_reval_section_slm.db"
for ext in ("", "-wal", "-shm"):
    p = sqlite_dir / f"fitz_reval_section_slm.db{ext}"
    if p.exists():
        p.unlink()
shutil.copy2(src, dst)
print(f"cloned {src.name} -> {dst.name}", flush=True)

summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
print("distilbart-cnn-12-6 loaded", flush=True)

conn = sqlite3.connect(str(dst))
rows = conn.execute(
    "SELECT id, content FROM krag_section_index WHERE content IS NOT NULL"
).fetchall()
print(f"resummarizing {len(rows)} sections...", flush=True)

t0 = time.monotonic()
ok = fail = skip = 0
for i, (sid, content) in enumerate(rows):
    text = (content or "").strip()
    if len(text) < 40:
        conn.execute("UPDATE krag_section_index SET summary = '' WHERE id = ?", (sid,))
        skip += 1
        continue
    try:
        out = summ(text[:3000], max_length=60, min_length=15, truncation=True)
        conn.execute(
            "UPDATE krag_section_index SET summary = ? WHERE id = ?",
            (out[0]["summary_text"].strip(), sid),
        )
        ok += 1
    except Exception:  # noqa: BLE001 — on failure, empty summary falls back to title
        conn.execute("UPDATE krag_section_index SET summary = '' WHERE id = ?", (sid,))
        fail += 1
    if (i + 1) % 50 == 0:
        conn.commit()
        print(f"  {i + 1}/{len(rows)}  ({time.monotonic() - t0:.0f}s)", flush=True)

conn.commit()
print(f"done: {ok} ok, {fail} failed, {skip} skipped ({time.monotonic() - t0:.0f}s)", flush=True)
