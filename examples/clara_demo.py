# examples/clara_demo.py
"""
CLaRa Engine Demo - Compression-Native RAG

This example demonstrates how to use Apple's CLaRa (Continuous Latent Reasoning)
engine for document compression and question answering.

Requirements:
    pip install fitz-ai[clara]
    # Or manually: pip install torch transformers accelerate bitsandbytes peft

Hardware:
    - GPU with 6GB+ VRAM (4-bit quantization)
    - GPU with 8GB+ VRAM (8-bit quantization)
    - GPU with 16GB+ VRAM (full precision)

Usage:
    python examples/clara_demo.py
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def run_clara_demo():
    """Run the CLaRa engine demo."""
    print("=" * 60)
    print("CLaRa Engine Demo - Compression-Native RAG")
    print("=" * 60)
    print()

    # Check for GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("WARNING: No GPU detected. CLaRa will run on CPU (very slow)")
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        return

    print()

    # Import CLaRa engine
    print("Loading CLaRa engine...")
    from fitz_ai.engines.clara import ClaraEngine
    from fitz_ai.engines.clara.config.schema import ClaraConfig, ClaraModelConfig
    from fitz_ai.core import Query

    # Configure with 4-bit quantization for lower VRAM usage
    config = ClaraConfig(
        model=ClaraModelConfig(
            model_name_or_path="apple/CLaRa-7B-Instruct/compression-16",
            variant="instruct",
            load_in_4bit=True,  # Use 4-bit for 8GB GPUs
        ),
    )

    # Initialize engine (downloads model on first run ~14GB)
    print("Initializing CLaRa engine (first run downloads ~14GB model)...")
    engine = ClaraEngine(config)
    print()

    # Sample documents about different topics
    documents = [
        """Python is a high-level, general-purpose programming language. Its design
        philosophy emphasizes code readability with the use of significant indentation.
        Python was created by Guido van Rossum and first released in 1991. It supports
        multiple programming paradigms including structured, object-oriented, and
        functional programming.""",
        """Machine learning is a subset of artificial intelligence that enables computers
        to learn from data and improve their performance without being explicitly
        programmed. Common types include supervised learning, unsupervised learning,
        and reinforcement learning. Deep learning is a subset of machine learning
        based on artificial neural networks.""",
        """The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in
        Paris, France. It was named after the engineer Gustave Eiffel, whose company
        designed and built the tower from 1887 to 1889. Standing 330 meters tall,
        it was the world's tallest structure until 1930.""",
        """Quantum computing is a type of computation that harnesses quantum mechanical
        phenomena such as superposition and entanglement. Unlike classical computers
        that use bits, quantum computers use quantum bits or qubits. This allows them
        to solve certain problems exponentially faster than classical computers.""",
    ]

    # Add documents
    print(f"Adding {len(documents)} documents to knowledge base...")
    engine.add_documents(documents)
    print()

    # Print knowledge base stats
    stats = engine.get_knowledge_stats()
    print(f"Knowledge base stats: {stats}")
    print()

    # Test questions
    questions = [
        "Who created Python and when was it released?",
        "What is machine learning?",
        "How tall is the Eiffel Tower?",
        "What are qubits?",
    ]

    print("=" * 60)
    print("Running queries...")
    print("=" * 60)
    print()

    for question in questions:
        print(f"Q: {question}")

        # Create query
        query = Query(text=question)

        # Get answer
        answer = engine.answer(query)

        print(f"A: {answer.text}")
        print(f"   [Sources: {len(answer.provenance)} docs used]")
        print()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)


def run_clara_direct():
    """
    Run CLaRa directly without the fitz-ai wrapper.

    This shows how to use CLaRa at the lowest level.
    """
    import os
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    print("=" * 60)
    print("Direct CLaRa Usage (without fitz-ai wrapper)")
    print("=" * 60)
    print()

    # Download model if needed
    from huggingface_hub import snapshot_download
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "fitz" / "clara"
    model_dir = cache_dir / "apple_CLaRa-7B-Instruct" / "compression-16"

    if not model_dir.exists():
        print("Downloading CLaRa model files...")
        snapshot_download(
            repo_id="apple/CLaRa-7B-Instruct",
            allow_patterns=["compression-16/*"],
            local_dir=str(cache_dir / "apple_CLaRa-7B-Instruct"),
        )

    # Add to path
    sys.path.insert(0, str(model_dir))

    # Import and load model
    print("Loading CLaRa model with 4-bit quantization...")
    from modeling_clara import CLaRa

    model = CLaRa.from_pretrained(
        str(model_dir),
        quantization="int4",
        device_map="auto",
    )

    # Test documents and question
    documents = [
        [
            "The Great Wall of China is over 13,000 miles long.",
            "Mount Everest is 29,032 feet tall, making it Earth's highest peak.",
            "The Amazon River is the largest river by discharge volume.",
        ]
    ]
    questions = ["What is the height of Mount Everest?"]

    # Set top-k
    model.generation_top_k = 3

    # Generate
    print("Generating answer...")
    answers = model.generate_from_text(
        questions=questions,
        documents=documents,
        max_new_tokens=64,
    )

    print()
    print(f"Question: {questions[0]}")
    print(f"Answer: {answers[0]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLaRa Engine Demo")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run CLaRa directly without fitz-ai wrapper",
    )
    args = parser.parse_args()

    if args.direct:
        run_clara_direct()
    else:
        run_clara_demo()
