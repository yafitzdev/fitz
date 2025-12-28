# fitz_ai/prompts/hierarchy/group_summary.py
"""
Prompt for generating group-level (Level 1) summaries.

Used to summarize all chunks from a single document/group.
Extracts: key themes, metrics, dates, notable points.

Version: v2 - Optimized for trend analysis and metric extraction
"""

PROMPT = """You are creating a summary for a knowledge retrieval system.
Your summary will be embedded and retrieved when users ask analytical questions.

TASK: Summarize this document to capture information useful for trend analysis and insights.

EXTRACT AND PRESERVE:
1. **Metrics & Numbers**: NPS scores, percentages, counts, response times, ratings
2. **Time References**: Dates, periods, "compared to last month", temporal markers
3. **Key Themes**: Main topics, recurring concerns, positive/negative patterns
4. **Changes & Trends**: Improvements, declines, shifts in priorities
5. **Notable Quotes**: Significant customer feedback or insights

FORMAT: Write 2-3 dense paragraphs. Be specific - include actual numbers and dates.
Do NOT use bullet points. Write in flowing prose optimized for retrieval."""
