# examples/02_tabular_sql.py
"""
Tabular Data - CSV files become queryable SQL tables.

Unlike other RAG systems that chunk CSVs into fragments, Fitz stores tables
natively in SQLite. This example uses optional synthesis so SQL results can be
formatted as computed answers.

Requirements:
    pip install fitz-sage
    Configure `synthesizer:` or pass a config_path with an endpoint-backed model.

Run:
    python examples/02_tabular_sql.py
"""

import shutil
import tempfile
from pathlib import Path

from fitz_sage import fitz

# =============================================================================
# Setup: Create sample CSV data
# =============================================================================

# Create a sample sales CSV
sample_csv = """product,region,quarter,revenue,units_sold
Widget A,North,Q1,15000,150
Widget A,South,Q1,12000,120
Widget A,North,Q2,18000,180
Widget A,South,Q2,14000,140
Widget B,North,Q1,25000,100
Widget B,South,Q1,22000,88
Widget B,North,Q2,28000,112
Widget B,South,Q2,24000,96
Widget C,North,Q1,8000,200
Widget C,South,Q1,9500,238
Widget C,North,Q2,10000,250
Widget C,South,Q2,11000,275
"""

# Write to temp file
temp_dir = Path(tempfile.mkdtemp())
csv_path = temp_dir / "sales_data.csv"
csv_path.write_text(sample_csv)
print(f"Created sample CSV at: {csv_path}\n")

# =============================================================================
# Step 1: Point Fitz at the CSV
# =============================================================================

f = fitz(collection="tabular_demo")

print("Pointing at CSV data...\n")

# =============================================================================
# Step 2: Ask natural language questions about the data
# =============================================================================

# Fitz automatically detects tabular queries and runs SQL
questions = [
    "What is the total revenue by region?",
    "Which product has the highest units sold?",
    "What was the average revenue per quarter?",
    "Compare Widget A vs Widget B total revenue",
    "Show me Q2 performance by product",
]

print("=" * 60)
print("TABULAR QUERIES")
print("=" * 60)

for i, question in enumerate(questions):
    print(f"\nQ: {question}")
    answer = f.query(question, source=str(temp_dir) if i == 0 else None)
    print(f"A: {answer.text}")

# =============================================================================
# Step 3: Mix tabular and document queries
# =============================================================================

# Add some documentation alongside the data
docs_path = temp_dir / "sales_guide.md"
docs_path.write_text(
    """
# Sales Performance Guide

## Regional Strategy
- North region focuses on enterprise clients with higher volume deals
- South region targets SMB with competitive pricing

## Product Lines
- Widget A: Entry-level product, high volume, lower margin
- Widget B: Premium product, lower volume, higher margin
- Widget C: Budget product, highest volume, lowest margin

## Q2 Goals
Target 15% growth over Q1 in both regions.
"""
)

print("\n" + "=" * 60)
print("HYBRID QUERIES (Data + Documents)")
print("=" * 60)

hybrid_questions = [
    "Did we hit our Q2 growth targets?",
    "Which product matches the 'premium, lower volume' strategy?",
    "What's the revenue breakdown for the budget product line?",
]

for i, question in enumerate(hybrid_questions):
    print(f"\nQ: {question}")
    answer = f.query(question, source=str(temp_dir) if i == 0 else None)
    print(f"A: {answer.text}")

# Cleanup
shutil.rmtree(temp_dir)
print("\n\nDemo complete! Temp files cleaned up.")
