# fitz/vector_db/loader.py

from __future__ import annotations

import os
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import httpx
from jinja2 import Template

from fitz.vector_db.base import SearchResult


class VectorDBSpec:
    """Parsed vector DB specification from YAML file."""

    def __init__(self, yaml_path: Path):
        with open(yaml_path) as f:
            self.spec = yaml.safe_load(f)

        self.name = self.spec['name']
        self.type = self.spec['type']
        self.description = self.spec.get('description', '')
        self.connection = self.spec['connection']
        self.operations = self.spec['operations']

    def is_local(self) -> bool:
        """Check if this is a local (non-HTTP) plugin."""
        return self.connection.get('type') == 'local'

    def get_local_class_path(self) -> Optional[str]:
        """Get Python class path for local implementations."""
        return self.operations.get('python_class')

    def build_base_url(self, **kwargs) -> str:
        """Build base URL from template and kwargs."""
        template = self.connection.get('base_url', '')

        # Apply defaults
        context = {}
        if 'default_host' in self.connection:
            context['host'] = kwargs.get('host', self.connection['default_host'])
        if 'default_port' in self.connection:
            context['port'] = kwargs.get('port', self.connection['default_port'])
        if 'default_environment' in self.connection:
            context['environment'] = kwargs.get('environment', self.connection['default_environment'])

        # Add all other kwargs
        context.update(kwargs)

        return Template(template).render(context)

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers from environment."""
        if 'auth' not in self.connection:
            return {}

        auth = self.connection['auth']

        # Optional auth - skip if not present
        if auth.get('optional') and not os.getenv(auth['env_var']):
            return {}

        env_var = auth['env_var']
        api_key = os.getenv(env_var)

        if not api_key:
            if not auth.get('optional'):
                raise ValueError(f"{env_var} not set")
            return {}

        if auth['type'] == 'bearer':
            header = auth.get('header', 'Authorization')
            scheme = auth.get('scheme', 'Bearer')
            return {header: f"{scheme} {api_key}"}
        elif auth['type'] == 'custom':
            return {auth['header']: api_key}

        return {}

    def render_template(self, template: Any, context: Dict[str, Any]) -> Any:
        """Render Jinja2 templates in values recursively."""
        if isinstance(template, str):
            if '{{' in template:
                return Template(template).render(context)
            return template
        elif isinstance(template, dict):
            return {k: self.render_template(v, context) for k, v in template.items()}
        elif isinstance(template, list):
            return [self.render_template(item, context) for item in template]
        else:
            return template

    def extract_value(self, data: Any, path: str) -> Any:
        """Extract value using dot notation path (e.g., 'result.collections')."""
        if not path:
            return data

        parts = path.split('.')
        current = data

        for part in parts:
            if part.isdigit():
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)

            if current is None:
                return None

        return current

    def transform_points(self, points: List[Dict], operation: str) -> List[Dict]:
        """Transform standard points format to provider-specific format."""
        op_spec = self.operations.get(operation, {})
        transform = op_spec.get('point_transform', {})

        # No transformation needed
        if transform.get('identity'):
            return points

        # Field mapping transformation
        transformed = []
        for point in points:
            new_point = {}
            for target_field, source_field in transform.items():
                if source_field in point:
                    new_point[target_field] = point[source_field]
            transformed.append(new_point)

        return transformed


class GenericVectorDBPlugin:
    """Generic vector DB plugin that executes YAML specifications."""

    plugin_type = "vector_db"

    def __init__(self, spec: VectorDBSpec, **kwargs):
        self.spec = spec
        self.plugin_name = spec.name
        self.kwargs = kwargs

        # Setup HTTP client
        base_url = spec.build_base_url(**kwargs)
        headers = spec.get_auth_headers()

        self.client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=30.0,
        )

        # Track vector dimension for auto-create
        self._vector_dim: Optional[int] = None

    def search(
            self,
            collection_name: str,
            query_vector: List[float],
            limit: int,
            with_payload: bool = True,
    ) -> List[SearchResult]:
        """Search for similar vectors in collection."""
        op = self.spec.operations['search']

        # Build context for template rendering
        context = {
            'collection': collection_name,
            'query_vector': query_vector,
            'limit': limit,
            'with_payload': with_payload,
            **self.kwargs,
        }

        # Build request
        endpoint = Template(op['endpoint']).render(context)
        body = self.spec.render_template(op.get('body', {}), context)

        # Send request
        response = self.client.request(
            method=op['method'],
            url=endpoint,
            json=body if body else None,
        )
        response.raise_for_status()

        # Extract results from response
        data = response.json()
        results_path = op['response']['results_path']
        results = self.spec.extract_value(data, results_path)

        if not results:
            return []

        # Map provider format to SearchResult
        mapping = op['response']['mapping']
        search_results = []

        for item in results:
            result_id = self.spec.extract_value(item, mapping['id'])
            result_score = self.spec.extract_value(item, mapping.get('score'))
            result_payload = self.spec.extract_value(item, mapping.get('payload', {}))

            search_result = SearchResult(
                id=str(result_id),
                score=float(result_score) if result_score is not None else None,
                payload=result_payload if result_payload else {},
            )
            search_results.append(search_result)

        return search_results

    def upsert(self, collection: str, points: List[Dict]) -> None:
        """Insert or update points in collection."""
        if not points:
            return

        # Detect vector dimension from first point
        if self._vector_dim is None:
            self._vector_dim = len(points[0]['vector'])

        op = self.spec.operations['upsert']

        # Auto-create collection if configured
        if op.get('auto_create_collection'):
            self._ensure_collection(collection, op)

        # Transform points to provider format
        transformed_points = self.spec.transform_points(points, 'upsert')

        # Build context
        context = {
            'collection': collection,
            'points': transformed_points,
            'vector_dim': self._vector_dim,
            **self.kwargs,
        }

        # Build request
        endpoint = Template(op['endpoint']).render(context)
        body = self.spec.render_template(op.get('body', {}), context)

        # Send request
        response = self.client.request(
            method=op['method'],
            url=endpoint,
            json=body,
        )
        response.raise_for_status()

    def _ensure_collection(self, collection: str, upsert_op: Dict):
        """Create collection if it doesn't exist (best-effort)."""
        try:
            endpoint = Template(upsert_op['create_collection_endpoint']).render(
                {'collection': collection, **self.kwargs}
            )

            body = self.spec.render_template(
                upsert_op.get('create_collection_body', {}),
                {'collection': collection, 'vector_dim': self._vector_dim, **self.kwargs}
            )

            response = self.client.request(
                method=upsert_op['create_collection_method'],
                url=endpoint,
                json=body if body else None,
            )

            # Ignore "already exists" errors (usually 409)
            if response.status_code not in [200, 201, 409]:
                response.raise_for_status()

        except Exception:
            # Best-effort - collection might already exist
            pass

    def delete_collection(self, collection: str) -> None:
        """Delete a collection."""
        if 'delete_collection' not in self.spec.operations:
            raise NotImplementedError(
                f"{self.plugin_name} does not support delete_collection"
            )

        op = self.spec.operations['delete_collection']
        context = {'collection': collection, **self.kwargs}

        endpoint = Template(op['endpoint']).render(context)

        response = self.client.request(
            method=op['method'],
            url=endpoint,
        )

        # Handle acceptable status codes
        success_codes = op.get('response', {}).get('success_codes', [200, 204])
        if response.status_code not in success_codes:
            response.raise_for_status()

    def list_collections(self) -> List[str]:
        """List all collections."""
        if 'list_collections' not in self.spec.operations:
            raise NotImplementedError(
                f"{self.plugin_name} does not support list_collections"
            )

        op = self.spec.operations['list_collections']
        context = {**self.kwargs}

        endpoint = Template(op['endpoint']).render(context)

        response = self.client.get(endpoint)
        response.raise_for_status()

        data = response.json()
        collections = self.spec.extract_value(data, op['response']['collections_path'])

        if not collections:
            return []

        # Extract collection names
        name_field = op['response'].get('name_field', 'name')
        return [c[name_field] if isinstance(c, dict) else c for c in collections]

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if 'get_stats' not in self.spec.operations:
            raise NotImplementedError(
                f"{self.plugin_name} does not support get_collection_stats"
            )

        op = self.spec.operations['get_stats']
        context = {'collection': collection, **self.kwargs}

        endpoint = Template(op['endpoint']).render(context)

        response = self.client.get(endpoint)
        response.raise_for_status()

        data = response.json()
        return self.spec.extract_value(data, op['response']['stats_path'])

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, 'client'):
            try:
                self.client.close()
            except Exception:
                pass


def load_vector_db_spec(plugin_name: str) -> VectorDBSpec:
    """Load vector DB specification from YAML file.

    Looks for YAML files in fitz/vector_db/plugins/ directory.
    """
    # Look for YAML file in plugins subdirectory
    plugins_dir = Path(__file__).parent / "plugins"
    yaml_path = plugins_dir / f"{plugin_name}.yaml"

    if not yaml_path.exists():
        raise ValueError(
            f"Vector DB plugin '{plugin_name}' not found. "
            f"Expected: {yaml_path}"
        )

    return VectorDBSpec(yaml_path)


def create_vector_db_plugin(plugin_name: str, **kwargs):
    """
    Create a vector DB plugin from YAML specification.

    For HTTP-based plugins: Returns GenericVectorDBPlugin
    For local plugins: Returns the specific Python implementation

    Args:
        plugin_name: Name of the plugin (e.g., 'qdrant', 'pinecone', 'local-faiss')
        **kwargs: Plugin-specific configuration (host, port, api_key, etc.)

    Returns:
        Vector DB plugin instance

    Examples:
        >>> db = create_vector_db_plugin('qdrant', host='localhost', port=6333)
        >>> db = create_vector_db_plugin('pinecone', index_name='my-index', project_id='abc')
        >>> db = create_vector_db_plugin('local-faiss', path='/tmp/vectors')
    """
    spec = load_vector_db_spec(plugin_name)

    # Handle local implementations (e.g., FAISS)
    if spec.is_local():
        class_path = spec.get_local_class_path()
        if not class_path:
            raise ValueError(
                f"Local plugin '{plugin_name}' missing python_class specification"
            )

        # Import and instantiate the Python class
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        PluginClass = getattr(module, class_name)

        return PluginClass(**kwargs)

    # HTTP-based plugins use generic implementation
    return GenericVectorDBPlugin(spec, **kwargs)