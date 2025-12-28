# fitz_ai/prompts/hierarchy/group_summary_epistemic.py
"""
Epistemic-aware prompt for generating group-level (Level 1) summaries.

Adapts the summarization prompt based on:
- Detected conflicts between sources
- Evidence density (sparse/moderate/dense)
- Agreement ratio

Produces summaries that honestly represent uncertainty and disagreement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_ai.ingest.enrichment.hierarchy.epistemic import EpistemicAssessment


# Base prompt for all summaries
BASE_PROMPT = """You are creating a summary for a knowledge retrieval system.
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


# Addendum for conflicting sources
CONFLICT_ADDENDUM = """
IMPORTANT - CONFLICTING SOURCES DETECTED:
The sources contain contradictory claims on: {conflict_topics}

You MUST:
- Acknowledge the disagreement explicitly in your summary
- Do NOT pick a side or resolve the conflict
- Present both perspectives: "Source A indicates X, while Source B suggests Y"
- If metrics differ, report the range: "NPS scores varied from X to Y"
- Flag uncertainty: "Sources disagree on..." or "Evidence is mixed regarding..."

Epistemic honesty is more important than a clean narrative."""


# Addendum for sparse evidence
SPARSE_EVIDENCE_ADDENDUM = """
IMPORTANT - LIMITED EVIDENCE:
This summary is based on only {chunk_count} sources ({evidence_density} evidence).

You MUST:
- Qualify claims appropriately: "Based on limited data..." or "The available evidence suggests..."
- Avoid overgeneralizing from few examples
- Do NOT claim trends or patterns that aren't clearly supported
- Be conservative - it's better to understate than overstate"""


# Addendum for low agreement
LOW_AGREEMENT_ADDENDUM = """
IMPORTANT - LOW SOURCE AGREEMENT:
Agreement ratio across sources is only {agreement_ratio:.0%}.

You MUST:
- Note the lack of consensus in your summary
- Present the range of perspectives rather than a unified view
- Use hedging language: "Some sources indicate...", "Views vary on..."
- Do NOT present contested claims as established facts"""


def build_epistemic_group_prompt(assessment: "EpistemicAssessment") -> str:
    """
    Build a group summary prompt adapted to the epistemic status.

    Args:
        assessment: Epistemic assessment of the chunk group

    Returns:
        Prompt string with appropriate epistemic guidance
    """
    prompt_parts = [BASE_PROMPT]

    # Add conflict guidance if needed
    if assessment.has_conflicts and assessment.conflict_topics:
        topics_str = ", ".join(assessment.conflict_topics)
        prompt_parts.append(CONFLICT_ADDENDUM.format(conflict_topics=topics_str))

    # Add sparse evidence guidance if needed
    if assessment.evidence_density == "sparse":
        prompt_parts.append(
            SPARSE_EVIDENCE_ADDENDUM.format(
                chunk_count=assessment.chunk_count,
                evidence_density=assessment.evidence_density,
            )
        )

    # Add low agreement guidance if needed
    if assessment.agreement_ratio < 0.7:
        prompt_parts.append(
            LOW_AGREEMENT_ADDENDUM.format(agreement_ratio=assessment.agreement_ratio)
        )

    return "\n".join(prompt_parts)


__all__ = ["build_epistemic_group_prompt"]
