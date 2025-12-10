from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.chat_client import CohereChatClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.embedding_client import CohereEmbeddingClient   # <-- you will create this file
from qdrant_client import QdrantClient
import os

client = QdrantClient(host="192.168.178.2", port=6333)
api_key = os.getenv("COHERE_API_KEY")

embedder = CohereEmbeddingClient(api_key)

retriever = RAGRetriever(
    client=client,
    embedder=embedder,
    collection="fitz_stack_test",
    top_k=15,
)

reranker = CohereRerankClient(api_key=api_key)
chat = CohereChatClient(api_key=api_key)

pipeline = RAGPipeline(
    retriever=retriever,
    reranker=reranker,
    chat_client=chat,
)

answer = pipeline.run("what are these docs about?")
print(answer)
