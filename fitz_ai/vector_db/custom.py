# fitz_ai/vector_db/custom.py
"""
Custom Vector Database Plugin.

Allows users to define their own vector DB endpoints in config.yaml.
No code changes needed - just configure the HTTP operations.

Example config.yaml:

    vector_db:
      plugin_name: custom
      kwargs:
        base_url: "http://localhost:8000"

        upsert:
          method: POST
          endpoint: "/collections/{collection}/points"
          body: '{"points": {points}}'

        search:
          method: POST
          endpoint: "/collections/{collection}/search"
          body: '{"vector": {query_vector}, "limit": {limit}}'
          results_path: "results"
          mapping:
            id: "id"
            score: "score"
            payload: "metadata"
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from fitz_ai.logging.logger import get_logger
from fitz_ai.vector_db.base import SearchResult

logger = get_logger(__name__)


def _substitute_vars(template: str, context: Dict[str, Any]) -> str:
    """
    Substitute {var} placeholders in a string.

    Handles both simple substitution and JSON serialization for complex types.
    """

    def replacer(match):
        var_name = match.group(1)
        if var_name not in context:
            return match.group(0)  # Keep original if not found

        value = context[var_name]

        # If it's a complex type, serialize to JSON
        if isinstance(value, (list, dict)):
            return json.dumps(value)

        return str(value)

    return re.sub(r'\{(\w+)\}', replacer, template)


def _extract_path(data: Any, path: str, default: Any = None) -> Any:
    """Extract value from nested dict using dot notation path."""
    if not path:
        return data

    parts = path.split('.')
    current = data

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            return default

    return current


class CustomVectorDB:
    """
    Custom Vector DB that reads operation definitions from kwargs.

    This allows users to connect to any HTTP-based vector database
    by defining the endpoints in their config.yaml.
    """

    plugin_name = "custom"
    plugin_type = "vector_db"

    def __init__(self, **kwargs):
        """
        Initialize custom vector DB.

        Required kwargs:
            base_url: Base URL for the API
            upsert: Upsert operation config
            search: Search operation config

        Optional kwargs:
            list_collections: List collections operation config
            get_stats: Get stats operation config
            delete_collection: Delete collection operation config
            auth: Authentication config
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = kwargs.get('base_url')
        if not self.base_url:
            raise ValueError("custom vector DB requires 'base_url' in kwargs")

        # Store operation configs
        self._upsert_config = kwargs.get('upsert')
        self._search_config = kwargs.get('search')
        self._list_config = kwargs.get('list_collections')
        self._stats_config = kwargs.get('get_stats')
        self._delete_config = kwargs.get('delete_collection')

        # Validate required operations
        if not self._upsert_config:
            raise ValueError("custom vector DB requires 'upsert' operation in kwargs")
        if not self._search_config:
            raise ValueError("custom vector DB requires 'search' operation in kwargs")

        # Build headers
        headers = {}
        auth_config = kwargs.get('auth')
        if auth_config:
            headers.update(self._build_auth_headers(auth_config))

        # Create HTTP client
        timeout = kwargs.get('timeout', 30)
        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

        logger.info(f"Custom vector DB initialized: {self.base_url}")

    def _build_auth_headers(self, auth_config: Dict) -> Dict[str, str]:
        """Build authentication headers from config."""
        header_name = auth_config.get('header', 'Authorization')

        # Get key from environment
        env_var = auth_config.get('value_env')
        if env_var:
            key = os.getenv(env_var)
            if not key:
                raise ValueError(f"Environment variable {env_var} not set")
        else:
            key = auth_config.get('value', '')

        # Apply format if specified
        format_str = auth_config.get('format', '{key}')
        header_value = format_str.replace('{key}', key)

        return {header_name: header_value}

    def _execute_operation(
            self,
            config: Dict,
            context: Dict[str, Any],
    ) -> httpx.Response:
        """Execute an HTTP operation based on config."""
        method = config.get('method', 'POST').upper()
        endpoint = _substitute_vars(config.get('endpoint', ''), context)

        # Build body if present
        body = None
        body_template = config.get('body')
        if body_template:
            body_str = _substitute_vars(body_template, context)
            try:
                body = json.loads(body_str)
            except json.JSONDecodeError:
                # If it's not valid JSON, send as-is
                body = body_str

        # Make request
        response = self.client.request(
            method=method,
            url=endpoint,
            json=body if isinstance(body, (dict, list)) else None,
            content=body if isinstance(body, str) else None,
        )

        return response

    # =========================================================================
    # Required Operations
    # =========================================================================

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        """Insert or update points in collection."""
        context = {
            'collection': collection,
            'points': points,
        }

        # If there's a point_transform, apply it
        transform = self._upsert_config.get('point_transform')
        if transform:
            points = self._transform_points(points, transform)
            context['points'] = points

        response = self._execute_operation(self._upsert_config, context)
        response.raise_for_status()

        logger.debug(f"Upserted {len(points)} points to {collection}")

    def search(
            self,
            collection_name: str,
            query_vector: List[float],
            limit: int,
            with_payload: bool = True,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        context = {
            'collection': collection_name,
            'query_vector': query_vector,
            'limit': limit,
            'with_payload': with_payload,
        }

        response = self._execute_operation(self._search_config, context)
        response.raise_for_status()

        data = response.json()

        # Extract results using configured path
        results_path = self._search_config.get('results_path', '')
        results = _extract_path(data, results_path, default=[])

        if not results:
            return []

        # Map results to SearchResult objects
        mapping = self._search_config.get('mapping', {})
        id_field = mapping.get('id', 'id')
        score_field = mapping.get('score', 'score')
        payload_field = mapping.get('payload', 'payload')

        search_results = []
        for item in results:
            result_id = _extract_path(item, id_field)
            result_score = _extract_path(item, score_field)
            result_payload = _extract_path(item, payload_field, default={})

            search_results.append(SearchResult(
                id=str(result_id) if result_id else "",
                score=float(result_score) if result_score is not None else None,
                payload=result_payload if isinstance(result_payload, dict) else {},
            ))

        return search_results

    def _transform_points(self, points: List[Dict], transform: Dict) -> List[Dict]:
        """Transform points format based on config."""
        transformed = []
        for point in points:
            new_point = {}
            for target_field, source_field in transform.items():
                if source_field in point:
                    new_point[target_field] = point[source_field]
            transformed.append(new_point)
        return transformed

    # =========================================================================
    # Optional Operations
    # =========================================================================

    def list_collections(self) -> List[str]:
        """List all collections."""
        if not self._list_config:
            raise NotImplementedError(
                "list_collections not configured. Add 'list_collections' to vector_db.kwargs"
            )

        response = self._execute_operation(self._list_config, {})
        response.raise_for_status()

        data = response.json()

        # Extract collections using configured path
        collections_path = self._list_config.get('collections_path', '')
        collections = _extract_path(data, collections_path, default=[])

        # Extract names if needed
        name_field = self._list_config.get('name_field')
        if name_field and collections:
            return [
                c[name_field] if isinstance(c, dict) else c
                for c in collections
            ]

        return list(collections)

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._stats_config:
            raise NotImplementedError(
                "get_collection_stats not configured. Add 'get_stats' to vector_db.kwargs"
            )

        context = {'collection': collection}
        response = self._execute_operation(self._stats_config, context)
        response.raise_for_status()

        data = response.json()

        # Extract stats using configured path
        stats_path = self._stats_config.get('stats_path', '')
        return _extract_path(data, stats_path, default={})

    def delete_collection(self, collection: str) -> None:
        """Delete a collection."""
        if not self._delete_config:
            raise NotImplementedError(
                "delete_collection not configured. Add 'delete_collection' to vector_db.kwargs"
            )

        context = {'collection': collection}
        response = self._execute_operation(self._delete_config, context)

        # Accept 404 as success (already deleted)
        if response.status_code not in (200, 204, 404):
            response.raise_for_status()

        logger.info(f"Deleted collection: {collection}")

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except Exception:
                pass


__all__ = ['CustomVectorDB']