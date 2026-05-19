# tools/retrieval_eval/_spike_smoketest.py
"""v-0-13-08 spike smoke test: distilbart-cnn-12-6 vs the LLM summaries in the DB."""
import sqlite3

from transformers import pipeline

from fitz_sage.core.paths import FitzPaths

db = FitzPaths.workspace() / "sqlite" / "fitz_reval_section.db"
c = sqlite3.connect(str(db))
rows = c.execute(
    "SELECT title, summary, content FROM krag_section_index "
    "WHERE content IS NOT NULL AND length(content) > 400 "
    "AND summary IS NOT NULL AND length(summary) > 20 LIMIT 6"
).fetchall()
print(f"pulled {len(rows)} sections with both content and an LLM summary")

print("loading sshleifer/distilbart-cnn-12-6 (downloads ~1.2GB on first run)...")
summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
print("model loaded.\n")

for title, llm_summary, content in rows:
    slm = summ(content[:3000], max_length=60, min_length=15, truncation=True)
    slm_text = slm[0]["summary_text"].strip()
    print("=" * 72)
    print("TITLE:", (title or "")[:90])
    print("LLM  :", (llm_summary or "").strip()[:320])
    print("SLM  :", slm_text[:320])
