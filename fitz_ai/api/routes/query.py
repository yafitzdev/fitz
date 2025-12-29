# fitz_ai/api/routes/query.py
"""Query and chat endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from fitz_ai.api.dependencies import get_fitz_instance
from fitz_ai.api.models.schemas import (
    ChatRequest,
    ChatResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Query the knowledge base.

    Submit a question and receive an answer with sources.
    """
    try:
        f = get_fitz_instance(request.collection)
        answer = f.ask(request.question, top_k=request.top_k)

        sources = [
            SourceInfo(
                source_id=p.source_id,
                excerpt=p.excerpt,
                metadata=p.metadata,
            )
            for p in answer.provenance
        ]

        return QueryResponse(
            text=answer.text,
            mode=answer.mode.value if answer.mode else None,
            sources=sources,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn chat with the knowledge base.

    Send a message along with conversation history. The server is stateless;
    the client is responsible for maintaining and sending the history.
    """
    try:
        # Build context-aware query from history + current message
        context_parts = []
        for msg in request.history:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_prefix}: {msg.content}")

        if context_parts:
            # Include history context in the query
            history_context = "\n".join(context_parts)
            full_query = (
                f"Previous conversation:\n{history_context}\n\nCurrent question: {request.message}"
            )
        else:
            full_query = request.message

        f = get_fitz_instance(request.collection)
        answer = f.ask(full_query, top_k=request.top_k)

        sources = [
            SourceInfo(
                source_id=p.source_id,
                excerpt=p.excerpt,
                metadata=p.metadata,
            )
            for p in answer.provenance
        ]

        return ChatResponse(
            text=answer.text,
            mode=answer.mode.value if answer.mode else None,
            sources=sources,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
