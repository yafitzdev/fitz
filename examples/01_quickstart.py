# examples/01_quickstart.py
"""
Quickstart - The simplest way to use Fitz.

This is the 90% use case: point at docs, ask questions, get governed evidence.

Requirements:
    pip install fitz-sage

Run:
    python examples/01_quickstart.py
"""

from fitz_sage import fitz

# =============================================================================
# Setup: Create a Fitz instance
# =============================================================================

f = fitz(collection="quickstart_demo")

# =============================================================================
# Step 1: Point Fitz at documents
# =============================================================================

# Point at any folder - Fitz handles PDFs, DOCX, Markdown, code, etc.
source = "./docs"  # Change to your docs folder
print(f"Pointing at documents in {source}...\n")

# =============================================================================
# Step 2: Ask questions
# =============================================================================

questions = [
    "What is this project about?",
    "How do I get started?",
    "What are the main features?",
]

for i, question in enumerate(questions):
    print(f"Q: {question}")
    pack = f.evidence(question, source=source if i == 0 else None)
    print(f"Mode: {pack.mode}\n")

    # Every evidence pack includes ranked source units.
    if pack.items:
        print("Evidence:")
        for item in pack.items[:3]:
            print(f"  - {item.file_path}: {item.excerpt[:120]}")
        print()

# =============================================================================
# That's it! For more control, see the other examples.
# =============================================================================
