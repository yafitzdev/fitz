from fitz.core.llm.embedding.plugins.local import LocalEmbeddingClient


def main() -> None:
    embedder = LocalEmbeddingClient()

    vec = embedder.embed("hello world")

    print("\n=== LOCAL EMBEDDING OUTPUT ===")
    print(f"Vector length: {len(vec)}")
    print(f"First 10 values: {vec[:10]}")
    print("=============================\n")


if __name__ == "__main__":
    main()
