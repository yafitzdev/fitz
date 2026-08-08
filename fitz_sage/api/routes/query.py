# fitz_sage/api/routes/query.py
"""Query and chat endpoints."""

from __future__ import annotations

from functools import partial

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from fitz_sage.api.dependencies import get_service
from fitz_sage.api.error_handlers import handle_api_errors
from fitz_sage.api.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EvidenceResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from fitz_sage.api.security import resolve_api_source
from fitz_sage.retrieval.rewriter.types import ConversationContext, ConversationMessage

router = APIRouter(tags=["query"])


def _to_conversation_context(history: list[ChatMessage]) -> ConversationContext | None:
    """Convert API history to ConversationContext for query rewriting."""
    if not history:
        return None
    messages = [ConversationMessage(role=msg.role, content=msg.content) for msg in history]
    return ConversationContext(history=messages)


@router.post("/answer", response_model=QueryResponse)
@handle_api_errors
async def answer(request: QueryRequest) -> QueryResponse:
    """
    Synthesize an answer from retrieved evidence.

    Submit a question and receive an answer with sources.
    Optionally include source to register documents before querying,
    or conversation_history for query rewriting.
    """
    service = get_service()
    collection = request.collection or "default"

    if request.source is not None:
        source = resolve_api_source(request.source)
        await run_in_threadpool(partial(service.point, source=source, collection=collection))

    context = _to_conversation_context(request.conversation_history)

    answer = await run_in_threadpool(
        partial(
            service.answer,
            question=request.question,
            collection=collection,
            conversation_context=context,
        )
    )

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
        metadata=answer.metadata,
    )


@router.post("/evidence", response_model=EvidenceResponse)
@handle_api_errors
async def evidence(request: QueryRequest) -> EvidenceResponse:
    """
    Retrieve governed evidence without answer synthesis.

    This is the retrieval-first endpoint: it returns ranked source units,
    Pyrrho mode/reasons, progressive-delivery metadata, and indexing status.
    """
    service = get_service()
    collection = request.collection or "default"

    if request.source is not None:
        source = resolve_api_source(request.source)
        await run_in_threadpool(partial(service.point, source=source, collection=collection))

    context = _to_conversation_context(request.conversation_history)

    pack = await run_in_threadpool(
        partial(
            service.evidence,
            question=request.question,
            collection=collection,
            conversation_context=context,
        )
    )

    return EvidenceResponse(**pack.to_dict())


@router.post("/chat", response_model=ChatResponse)
@handle_api_errors
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Multi-turn chat with the knowledge base.

    Send a message along with conversation history. The server is stateless;
    the client is responsible for maintaining and sending the history.

    A configured query-intelligence provider may rewrite conversational
    references using this history. The deterministic default uses the current
    message as written.
    """
    service = get_service()
    context = _to_conversation_context(request.history)

    answer = await run_in_threadpool(
        partial(
            service.answer,
            question=request.message,
            collection=request.collection or "default",
            conversation_context=context,
        )
    )

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
        metadata=answer.metadata,
    )
