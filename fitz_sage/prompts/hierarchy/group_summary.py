# fitz_sage/prompts/hierarchy/group_summary.py
"""
Prompt for generating group-level (Level 1) summaries.

Used to summarize all chunks from a single document/group.
Extracts: key themes, metrics, dates, notable points.

Version: v3 - Adapts to code vs documentation content
"""

PROMPT = """You are creating a summary for a knowledge retrieval system.
Your summary will be stored and retrieved when users ask analytical questions.

TASK: Summarize this content to capture information useful for understanding and insights.

ADAPT TO CONTENT TYPE:

**If this is CODE (functions, classes, modules):**
- What does this code do? What problem does it solve?
- Key components: main classes, functions, their purposes
- How do the pieces fit together? What's the architecture?
- Dependencies and relationships between components

**If this is DOCUMENTATION or DATA:**
- Metrics & Numbers: scores, percentages, counts, ratings
- Time References: dates, periods, temporal markers
- Key Themes: main topics, patterns, concerns
- Changes & Trends: improvements, declines, shifts

FORMAT: Write 2-3 dense paragraphs. Be specific - include actual names, numbers, and details.
Do NOT use bullet points. Write in flowing prose optimized for retrieval."""
