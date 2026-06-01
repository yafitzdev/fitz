# examples/01_quickstart.py
"""
Quickstart - The simplest way to use Fitz.

This is the 90% use case: point at docs, ask questions, get answers with sources.

Requirements:
    pip install fitz-sage
    export OPENAI_API_KEY="your-key"  # if using a hosted OpenAI-compatible endpoint

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
    answer = f.query(question, source=source if i == 0 else None)
    print(f"A: {answer.text}\n")

    # Every answer includes sources
    if answer.provenance:
        print("Sources:")
        for source in answer.provenance[:3]:  # Show top 3
            print(f"  - {source.source_id}")
        print()

# =============================================================================
# That's it! For more control, see the other examples.
# =============================================================================
