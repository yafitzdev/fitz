# fitz_sage/api/routes/collections.py
"""Collection management endpoints."""

from __future__ import annotations

from functools import partial

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from fitz_sage.api.dependencies import get_service
from fitz_sage.api.error_handlers import handle_api_errors
from fitz_sage.api.models.schemas import (
    CollectionInfo,
    CollectionStats,
    IndexingStatus,
    IngestRequest,
)
from fitz_sage.api.security import resolve_api_source

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=list[CollectionInfo])
@handle_api_errors
async def list_collections() -> list[CollectionInfo]:
    """List all available collections."""
    service = get_service()
    collections = await run_in_threadpool(service.list_collections)

    return [CollectionInfo(name=c.name, item_count=c.item_count) for c in collections]


@router.get("/{name}", response_model=CollectionStats)
@handle_api_errors
async def get_collection(name: str) -> CollectionStats:
    """
    Get statistics for a specific collection.
    """
    service = get_service()
    info = await run_in_threadpool(service.get_collection, name)

    return CollectionStats(
        name=info.name,
        item_count=info.item_count,
        metadata=info.metadata,
    )


@router.delete("/{name}")
@handle_api_errors
async def delete_collection(name: str) -> dict:
    """
    Delete a collection.

    Returns whether the collection was deleted.
    """
    service = get_service()
    deleted = await run_in_threadpool(service.delete_collection, name)

    return {"deleted": deleted, "collection": name}


@router.post("/{name}/documents", response_model=IndexingStatus)
@handle_api_errors
async def ingest_documents(name: str, request: IngestRequest) -> IndexingStatus:
    """
    Register documents into a collection.

    Source indexing completes before this endpoint returns. Optional model-backed
    enrichment may continue afterward.
    """
    service = get_service()
    source = resolve_api_source(request.source)
    await run_in_threadpool(partial(service.point, source=source, collection=name))
    status = await run_in_threadpool(service.indexing_status, name)
    return IndexingStatus(**status)


@router.get("/{name}/status", response_model=IndexingStatus)
@handle_api_errors
async def collection_status(name: str) -> IndexingStatus:
    """Report source-index health and optional enrichment progress."""
    service = get_service()
    status = await run_in_threadpool(service.indexing_status, name)
    return IndexingStatus(**status)
