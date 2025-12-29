# fitz_ai/api/routes/ingest.py
"""Document ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from fitz_ai.api.dependencies import get_fitz_instance
from fitz_ai.api.models.schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest documents into the knowledge base.

    Provide a path to a file or directory. Documents will be chunked,
    embedded, and stored in the vector database.
    """
    try:
        f = get_fitz_instance(request.collection)
        stats = f.ingest(request.source, clear_existing=request.clear_existing)

        return IngestResponse(
            documents=stats.documents,
            chunks=stats.chunks,
            collection=stats.collection,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
