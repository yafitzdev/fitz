# examples/05_advanced_queries.py
"""
Advanced Queries - Keyword matching, governance, and query patterns.

Fitz has built-in intelligence for different query types:
- Exact keyword matching (IDs, tickets, versions)
- Structural + lexical broad recall
- Aggregation queries (trends, summaries)
- Comparison queries (A vs B)

Requirements:
    pip install fitz-sage

Run:
    python examples/05_advanced_queries.py
"""

import tempfile
from pathlib import Path

from fitz_sage import fitz


def show_pack(pack, max_items: int = 2) -> None:
    """Print a compact EvidencePack preview."""
    print(f"Mode: {pack.mode}")
    for item in pack.items[:max_items]:
        print(f"  - {item.file_path}: {item.excerpt[:140]}")
    print()


# =============================================================================
# Setup: Create documents with identifiers
# =============================================================================

temp_dir = Path(tempfile.mkdtemp())

# Bug reports with IDs
(temp_dir / "bugs.md").write_text(
    """
# Bug Reports

## BUG-1001: Login timeout on slow connections
Status: Fixed in v2.3.1
Users on connections slower than 1Mbps experienced login timeouts.
Root cause: Hardcoded 5-second timeout.
Fix: Made timeout configurable, default increased to 30s.

## BUG-1002: Dashboard charts not rendering
Status: Open
Charts fail to render when dataset exceeds 10,000 points.
Investigating performance optimization options.

## BUG-1003: Email notifications delayed
Status: Fixed in v2.3.2
Notifications were batched hourly instead of sent immediately.
Fix: Implemented real-time notification queue.
"""
)

# Release notes with versions
(temp_dir / "releases.md").write_text(
    """
# Release Notes

## v2.3.2 (2024-01-15)
- Fixed: Email notification delays (BUG-1003)
- Improved: Dashboard loading performance
- Added: Export to CSV feature

## v2.3.1 (2024-01-08)
- Fixed: Login timeout issues (BUG-1001)
- Fixed: Password reset email formatting
- Security: Updated dependencies

## v2.3.0 (2024-01-01)
- New: Dark mode support
- New: Custom dashboard layouts
- Improved: Search performance by 40%
"""
)

# Feature comparison
(temp_dir / "plans.md").write_text(
    """
# Pricing Plans

## Free Plan
- Up to 3 users
- 1GB storage
- Email support
- Basic analytics

## Pro Plan ($29/month)
- Up to 25 users
- 50GB storage
- Priority support
- Advanced analytics
- API access

## Enterprise Plan (Custom)
- Unlimited users
- Unlimited storage
- 24/7 dedicated support
- Custom integrations
- SLA guarantee
"""
)

# =============================================================================
# Step 1: Point Fitz at documents
# =============================================================================

f = fitz(collection="advanced_demo")
print("Pointing at documents...\n")

# =============================================================================
# Step 2: Exact keyword matching
# =============================================================================

print("=" * 60)
print("EXACT KEYWORD MATCHING")
print("=" * 60)
print("Fitz detects identifiers (BUG-XXXX, vX.X.X) and filters precisely.\n")

keyword_queries = [
    "What is BUG-1001?",
    "What was fixed in v2.3.1?",
    "Tell me about BUG-1003",
]

for i, q in enumerate(keyword_queries):
    print(f"Q: {q}")
    pack = f.evidence(q, source=str(temp_dir) if i == 0 else None)
    show_pack(pack)

# =============================================================================
# Step 3: Comparison queries
# =============================================================================

print("=" * 60)
print("COMPARISON QUERIES")
print("=" * 60)
print("Fitz detects 'vs', 'compare', 'difference' and retrieves both entities.\n")

comparison_queries = [
    "Compare Free vs Pro plan",
    "What's the difference between v2.3.1 and v2.3.2?",
    "Pro plan vs Enterprise - what do I get?",
]

for q in comparison_queries:
    print(f"Q: {q}")
    pack = f.evidence(q)
    show_pack(pack)

# =============================================================================
# Step 4: Aggregation queries
# =============================================================================

print("=" * 60)
print("AGGREGATION QUERIES")
print("=" * 60)
print("Questions about trends, lists, or summaries use hierarchical context.\n")

aggregation_queries = [
    "List all open bugs",
    "What features were added in January 2024?",
    "Summarize the recent releases",
]

for q in aggregation_queries:
    print(f"Q: {q}")
    pack = f.evidence(q)
    show_pack(pack)

# =============================================================================
# Step 5: Honest "I don't know" responses
# =============================================================================

print("=" * 60)
print("EPISTEMIC HONESTY")
print("=" * 60)
print("Fitz admits when information isn't in the documents.\n")

unknown_queries = [
    "What was fixed in v3.0.0?",  # Doesn't exist
    "What is BUG-9999?",  # Doesn't exist
    "What's the pricing for the Team plan?",  # Doesn't exist
]

for q in unknown_queries:
    print(f"Q: {q}")
    pack = f.evidence(q)
    show_pack(pack)

# =============================================================================
# Step 6: Working with EvidencePack objects
# =============================================================================

print("=" * 60)
print("EVIDENCE PACK DETAILS")
print("=" * 60)

pack = f.evidence("What bugs were fixed in v2.3.2?")

print(f"Query: {pack.query}")
print(f"Mode: {pack.mode}")
print(f"\nEvidence ({len(pack.items)} item(s)):")
for item in pack.items:
    print(f"  - {item.file_path}")
    print(f"    Excerpt: {item.excerpt[:100]}...")

print(f"\nMetadata keys: {list(pack.metadata.keys())}")

# Cleanup
import shutil

shutil.rmtree(temp_dir)
print("\n\nDemo complete!")
