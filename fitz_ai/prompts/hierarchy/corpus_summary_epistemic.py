# fitz_ai/prompts/hierarchy/corpus_summary_epistemic.py
"""
Epistemic-aware prompt for generating corpus-level (Level 2) summaries.

Synthesizes insights across group summaries while:
- Propagating uncertainty from lower levels
- Noting cross-group conflicts
- Maintaining epistemic honesty at the highest abstraction level
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fitz_ai.ingest.enrichment.hierarchy.epistemic import EpistemicAssessment


# Base prompt for corpus synthesis
BASE_PROMPT = """You are synthesizing multiple document summaries into a corpus-level overview.
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


# Addendum when group summaries contain conflicts
GROUPS_WITH_CONFLICTS_ADDENDUM = """
IMPORTANT - SOME GROUP SUMMARIES CONTAIN CONFLICTS:
{conflict_count} of {total_groups} group summaries noted internal contradictions.
Contested topics include: {all_conflict_topics}

You MUST:
- Propagate this uncertainty to the corpus level
- Do NOT present a unified trend if underlying data is contested
- Note which conclusions are well-supported vs which are uncertain
- Use phrases like: "While some sources agree on X, there is disagreement about Y"
- If groups disagree with each other, highlight the cross-group conflict"""


# Addendum when evidence is sparse across groups
SPARSE_CORPUS_ADDENDUM = """
IMPORTANT - CORPUS HAS LIMITED COVERAGE:
This synthesis is based on {total_groups} groups with {total_evidence_density} overall evidence.

You MUST:
- Be conservative about claiming corpus-wide trends
- Distinguish between patterns with strong support vs weak support
- Avoid extrapolating beyond what the data shows
- Acknowledge gaps: "The available data does not address..."
- Prefer "appears to" over "is" for uncertain patterns"""


# Addendum for cross-group disagreement
CROSS_GROUP_CONFLICT_ADDENDUM = """
IMPORTANT - GROUPS SHOW CONFLICTING PATTERNS:
Different document groups present conflicting conclusions.

You MUST:
- Highlight the disagreement at the corpus level
- Present multiple interpretations where relevant
- Do NOT force a false consensus
- Consider whether conflicts reflect real variation or data quality issues
- Phrase as: "Group A shows X trend, while Group B shows the opposite pattern of Y"
"""


def build_epistemic_corpus_prompt(
    group_assessments: list["EpistemicAssessment"],
) -> str:
    """
    Build a corpus summary prompt adapted to the epistemic status of groups.

    Args:
        group_assessments: Epistemic assessments from each group summary

    Returns:
        Prompt string with appropriate epistemic guidance
    """
    prompt_parts = [BASE_PROMPT]

    if not group_assessments:
        return BASE_PROMPT

    # Count groups with conflicts
    groups_with_conflicts = [a for a in group_assessments if a.has_conflicts]
    conflict_count = len(groups_with_conflicts)
    total_groups = len(group_assessments)

    # Gather all conflict topics
    all_conflict_topics: set[str] = set()
    for assessment in groups_with_conflicts:
        all_conflict_topics.update(assessment.conflict_topics)

    # Add conflict propagation guidance if needed
    if conflict_count > 0:
        topics_str = ", ".join(sorted(all_conflict_topics)) if all_conflict_topics else "various topics"
        prompt_parts.append(
            GROUPS_WITH_CONFLICTS_ADDENDUM.format(
                conflict_count=conflict_count,
                total_groups=total_groups,
                all_conflict_topics=topics_str,
            )
        )

    # Assess overall evidence density
    sparse_count = sum(1 for a in group_assessments if a.evidence_density == "sparse")
    if sparse_count > total_groups / 2:
        total_evidence_density = "sparse"
    elif sparse_count > 0:
        total_evidence_density = "mixed"
    else:
        total_evidence_density = "adequate"

    if total_evidence_density in ("sparse", "mixed"):
        prompt_parts.append(
            SPARSE_CORPUS_ADDENDUM.format(
                total_groups=total_groups,
                total_evidence_density=total_evidence_density,
            )
        )

    # Check for low agreement across groups
    avg_agreement = sum(a.agreement_ratio for a in group_assessments) / len(group_assessments)
    if avg_agreement < 0.6:
        prompt_parts.append(CROSS_GROUP_CONFLICT_ADDENDUM)

    return "\n".join(prompt_parts)


def build_epistemic_corpus_context(
    group_assessments: list["EpistemicAssessment"],
) -> str:
    """
    Build a context block summarizing epistemic status of input groups.

    This is prepended to the group summaries to give the LLM visibility
    into the confidence level of each group.

    Args:
        group_assessments: Epistemic assessments from each group

    Returns:
        Context string describing group epistemic status
    """
    if not group_assessments:
        return ""

    lines = ["EPISTEMIC STATUS OF INPUT GROUPS:"]

    for i, assessment in enumerate(group_assessments, 1):
        status_parts = []

        # Evidence density
        status_parts.append(f"evidence={assessment.evidence_density}")

        # Conflicts
        if assessment.has_conflicts:
            topics = ", ".join(assessment.conflict_topics[:2])  # Limit to 2 topics
            status_parts.append(f"conflicts on: {topics}")
        else:
            status_parts.append("no conflicts")

        # Agreement
        if assessment.agreement_ratio < 0.7:
            status_parts.append(f"agreement={assessment.agreement_ratio:.0%}")

        lines.append(f"  Group {i}: {'; '.join(status_parts)}")

    lines.append("")  # Empty line before content

    return "\n".join(lines)


__all__ = ["build_epistemic_corpus_prompt", "build_epistemic_corpus_context"]
