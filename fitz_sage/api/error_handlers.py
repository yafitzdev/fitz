# fitz_sage/api/error_handlers.py
"""
API error handling utilities.

Provides a decorator to standardize exception handling across all API routes,
reducing boilerplate and ensuring consistent error responses.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Callable, TypeVar

from fastapi import HTTPException

from fitz_sage.services.fitz_service import CollectionNotFoundError, FitzServiceError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def handle_api_errors(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for standardized API error handling.

    Maps common exceptions to appropriate HTTP status codes:
    - ValueError -> 400 Bad Request
    - KeyError -> 404 Not Found
    - FileNotFoundError -> 404 Not Found
    - HTTPException -> Re-raised as-is
    - Exception -> 500 Internal Server Error

    Usage:
        @router.get("/{id}")
        @handle_api_errors
        async def get_item(id: str):
            return service.get(id)
    """

    @wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except HTTPException:
            # Re-raise FastAPI HTTPExceptions as-is
            raise
        except CollectionNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            # Client error - bad input
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError as e:
            # Resource not found
            detail = str(e).strip("'\"") if str(e) else "Resource not found"
            raise HTTPException(status_code=404, detail=detail)
        except FileNotFoundError as e:
            # File/resource not found
            raise HTTPException(status_code=404, detail=str(e))
        except PermissionError as e:
            # Access denied
            raise HTTPException(status_code=403, detail=str(e))
        except NotImplementedError as e:
            # Feature not implemented
            raise HTTPException(status_code=501, detail=str(e))
        except FitzServiceError as e:
            logger.exception(f"Service error in {fn.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Fitz service operation failed")
        except Exception as e:
            # Unexpected error - log and return 500
            logger.exception(f"Unexpected error in {fn.__name__}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return wrapper


__all__ = ["handle_api_errors"]
