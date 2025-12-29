# fitz_ai/api/routes/collections.py
"""Collection management endpoints."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from fitz_ai.api.dependencies import get_vector_db
from fitz_ai.api.models.schemas import CollectionInfo, CollectionStats

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=List[CollectionInfo])
async def list_collections() -> List[CollectionInfo]:
    """
    List all available collections.

    Returns collection names and chunk counts.
    """
    try:
        vdb = get_vector_db()

        if not hasattr(vdb, "list_collections"):
            raise HTTPException(
                status_code=501,
                detail="Vector DB plugin does not support listing collections",
            )

        collection_names = vdb.list_collections()

        result = []
        for name in collection_names:
            chunk_count = 0
            if hasattr(vdb, "get_collection_stats"):
                try:
                    stats = vdb.get_collection_stats(name)
                    chunk_count = stats.get("count", stats.get("chunk_count", 0))
                except Exception:
                    pass
            result.append(CollectionInfo(name=name, chunk_count=chunk_count))

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{name}", response_model=CollectionStats)
async def get_collection(name: str) -> CollectionStats:
    """
    Get statistics for a specific collection.
    """
    try:
        vdb = get_vector_db()

        if not hasattr(vdb, "get_collection_stats"):
            raise HTTPException(
                status_code=501,
                detail="Vector DB plugin does not support collection stats",
            )

        stats = vdb.get_collection_stats(name)

        return CollectionStats(
            name=name,
            chunk_count=stats.get("count", stats.get("chunk_count", 0)),
            metadata=stats,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
async def delete_collection(name: str) -> dict:
    """
    Delete a collection.

    Returns the number of chunks deleted.
    """
    try:
        vdb = get_vector_db()

        if not hasattr(vdb, "delete_collection"):
            raise HTTPException(
                status_code=501,
                detail="Vector DB plugin does not support deleting collections",
            )

        result = vdb.delete_collection(name)

        # Some plugins return count, others return None
        deleted_count = result if isinstance(result, int) else 0

        return {"deleted": True, "collection": name, "chunks_deleted": deleted_count}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
