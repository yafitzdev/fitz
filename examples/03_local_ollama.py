# examples/03_local_ollama.py
"""
Fully Local Setup - No API keys, no cloud, complete privacy.

Fitz can run 100% locally using:
- Ollama for the LLM (via its OpenAI-compatible endpoint)
- SQLite for storage

Requirements:
    1. Install Ollama: https://ollama.ai
    2. Pull a model:
       ollama pull llama3.2
    3. Start Ollama:
       ollama serve

Run:
    python examples/03_local_ollama.py
"""

import tempfile
from pathlib import Path

# =============================================================================
# Step 1: Create local config
# =============================================================================

# Fitz auto-detects Ollama, but you can also configure manually
config_content = """
# Local-only configuration - no API keys needed.
# Point fitz-sage at Ollama's OpenAI-compatible endpoint.

chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: http://localhost:11434/v1
chat_smart_model: llama3.2

collection: local_demo
"""

# Write config to temp location
temp_dir = Path(tempfile.mkdtemp())
config_path = temp_dir / "config.yaml"
config_path.write_text(config_content)

# =============================================================================
# Step 2: Use Fitz with local config
# =============================================================================

from fitz_sage import fitz

# Create instance with local config
f = fitz(collection="local_demo", config_path=config_path)

# Create some test documents
docs_dir = temp_dir / "docs"
docs_dir.mkdir()

(docs_dir / "privacy.md").write_text(
    """
# Data Privacy Policy

All data processing happens locally on your machine.
No data is sent to external servers.
Documents are stored in a local SQLite database.
Answers are generated using local Ollama models.
"""
)

(docs_dir / "setup.md").write_text(
    """
# Local Setup Guide

1. Install Ollama from ollama.ai
2. Pull a model: llama3.2
3. Run 'ollama serve' to start the local server
4. Fitz stores everything in a local SQLite database
"""
)

# =============================================================================
# Step 3: Point at docs and query locally
# =============================================================================

print("Pointing at documents locally...\n")

try:
    # Ask questions - everything runs locally
    print("Q: How is my data protected?")
    answer = f.query("How is my data protected?", source=str(docs_dir))
    print(f"A: {answer.text}\n")

    print("Q: What do I need to set up for local usage?")
    answer = f.query("What do I need to set up for local usage?")
    print(f"A: {answer.text}\n")

    print("=" * 60)
    print("SUCCESS! Everything ran locally:")
    print("  - LLM: Ollama (llama3.2)")
    print("  - Storage: SQLite (local file)")
    print("  - No data left your machine")
    print("=" * 60)

except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure Ollama is running:")
    print("  1. Install from https://ollama.ai")
    print("  2. Run: ollama pull llama3.2")
    print("  3. Run: ollama serve")

# Cleanup
import shutil

shutil.rmtree(temp_dir)
