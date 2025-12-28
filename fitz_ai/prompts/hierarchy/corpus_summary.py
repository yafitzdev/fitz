# fitz_ai/prompts/hierarchy/corpus_summary.py
"""
Prompt for generating corpus-level (Level 0) summaries.

Used to synthesize insights across all group summaries.
Identifies: patterns over time, evolution, cross-cutting themes.

Version: v2 - Optimized for trend analysis and metric extraction
"""

PROMPT = """You are synthesizing multiple document summaries into a corpus-level overview.
This overview will be retrieved when users ask about trends, patterns, or evolution over time.

TASK: Identify patterns and trends across all the documents.

SYNTHESIZE:
1. **Temporal Patterns**: How metrics/themes evolved over time (if dates present)
2. **Consistent Themes**: Topics that appear across multiple documents
3. **Progression**: What improved, what declined, what emerged as priorities
4. **Key Metrics Journey**: Track how specific numbers changed (e.g., NPS: 42 -> 51 -> 64)
5. **Strategic Insights**: What do these patterns suggest for decision-making

FORMAT: Write 3-4 paragraphs. Be specific about progression and cite which documents
support each insight. Optimize for answering "What are the trends?" style questions."""
