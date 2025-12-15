from fitz.core.llm.rerank.plugins.local import LocalRerankClient
from fitz.core.models.chunk import Chunk


def main() -> None:
    reranker = LocalRerankClient()

    chunks = [
        Chunk(
            id="c1",
            doc_id="d1",
            chunk_index=0,
            content="The cat sat on the mat.",
            metadata={},
        ),
        Chunk(
            id="c2",
            doc_id="d1",
            chunk_index=1,
            content="Quantum mechanics describes subatomic particles.",
            metadata={},
        ),
        Chunk(
            id="c3",
            doc_id="d1",
            chunk_index=2,
            content="Cats are small domesticated mammals.",
            metadata={},
        ),
    ]

    ranked = reranker.rerank(
        query="What is a cat?",
        chunks=chunks,
    )

    print("\n=== LOCAL RERANK OUTPUT ===")
    for i, c in enumerate(ranked, start=1):
        print(f"{i}. {c.id}: {c.content}")
    print("==========================\n")


if __name__ == "__main__":
    main()
