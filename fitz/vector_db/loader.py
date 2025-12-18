# fitz/vector_db/loader.py
"""
Vector DB plugin loader with YAML specifications.

Auto-detects connection details from fitz.core.detect (single source of truth).
"""

from __future__ import annotations

import importlib
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from jinja2 import Template

from fitz.core.utils import extract_path
from fitz.vector_db.base import SearchResult


def _string_to_uuid(s: str) -> str:
    """
    Convert an arbitrary string ID to a deterministic UUID.

    Uses UUID5 with a fixed namespace for determinism - same input
    always produces the same UUID.
    """
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, s))


class VectorDBSpec:
    """Parsed vector DB specification from YAML file."""

    def __init__(self, yaml_path: Path):
        with open(yaml_path) as f:
            self.spec = yaml.safe_load(f)

        self.name = self.spec["name"]
        self.type = self.spec["type"]
        self.description = self.spec.get("description", "")
        self.connection = self.spec["connection"]
        self.operations = self.spec["operations"]
        self.features = self.spec.get("features", {})

    def is_local(self) -> bool:
        """Check if this is a local (non-HTTP) plugin."""
        return self.connection.get("type") == "local"

    def get_local_class_path(self) -> Optional[str]:
        """Get Python class path for local implementations."""
        return self.operations.get("python_class")

    def requires_uuid_ids(self) -> bool:
        """Check if this vector DB requires UUID IDs (from YAML spec)."""
        return self.features.get("requires_uuid_ids", False)

    def get_auto_detect_service(self) -> Optional[str]:
        """Get the service name for auto-detection (from YAML spec)."""
        return self.features.get("auto_detect")

    def build_base_url(self, **kwargs) -> str:
        """Build base URL from template and kwargs."""
        template = self.connection.get("base_url", "")

        # Apply defaults from YAML spec
        context = {}
        if "default_host" in self.connection:
            context["host"] = kwargs.get("host", self.connection["default_host"])
        if "default_port" in self.connection:
            context["port"] = kwargs.get("port", self.connection["default_port"])
        if "default_environment" in self.connection:
            context["environment"] = kwargs.get(
                "environment", self.connection["default_environment"]
            )

        # Add all other kwargs
        context.update(kwargs)

        return Template(template).render(context)

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers from environment."""
        if "auth" not in self.connection:
            return {}

        auth = self.connection["auth"]

        # Optional auth - skip if not present
        if auth.get("optional") and not os.getenv(auth["env_var"]):
            return {}

        env_var = auth["env_var"]
        api_key = os.getenv(env_var)

        if not api_key:
            if not auth.get("optional"):
                raise ValueError(f"{env_var} not set")
            return {}

        if auth["type"] == "bearer":
            header = auth.get("header", "Authorization")
            scheme = auth.get("scheme", "Bearer")
            return {header: f"{scheme} {api_key}"}
        elif auth["type"] == "custom":
            return {auth["header"]: api_key}

        return {}

    def render_template(self, template: Any, context: Dict[str, Any]) -> Any:
        """Render Jinja2 templates in values recursively."""
        if isinstance(template, str):
            if "{{" in template:
                stripped = template.strip()
                if stripped.startswith("{{") and stripped.endswith("}}"):
                    var_name = stripped[2:-2].strip()
                    if var_name in context:
                        return context[var_name]
                return Template(template).render(context)
            return template
        elif isinstance(template, dict):
            return {k: self.render_template(v, context) for k, v in template.items()}
        elif isinstance(template, list):
            return [self.render_template(item, context) for item in template]
        else:
            return template

    def transform_points(self, points: List[Dict], operation: str) -> List[Dict]:
        """Transform standard points format to provider-specific format."""
        op_spec = self.operations.get(operation, {})
        transform = op_spec.get("point_transform", {})

        if transform.get("identity"):
            return points

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

        base_url = spec.build_base_url(**kwargs)
        headers = spec.get_auth_headers()

        self.client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=30.0,
        )

        self._vector_dim: Optional[int] = None

    def _convert_point_ids(self, points: List[Dict]) -> List[Dict]:
        """Convert string IDs to UUIDs if required by the vector DB (from YAML spec)."""
        if not self.spec.requires_uuid_ids():
            return points

        converted = []
        for point in points:
            new_point = dict(point)
            original_id = point.get("id")

            if original_id is not None and isinstance(original_id, str):
                try:
                    uuid.UUID(original_id)
                except ValueError:
                    new_point["id"] = _string_to_uuid(original_id)
                    if "payload" not in new_point:
                        new_point["payload"] = {}
                    new_point["payload"]["_original_id"] = original_id

            converted.append(new_point)

        return converted

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        with_payload: bool = True,
    ) -> List[SearchResult]:
        """Search for similar vectors in collection."""
        if "search" not in self.spec.operations:
            raise NotImplementedError(f"{self.plugin_name} does not support search")

        op = self.spec.operations["search"]

        context = {
            "collection": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": with_payload,
            **self.kwargs,
        }

        endpoint = Template(op["endpoint"]).render(context)
        body = self.spec.render_template(op.get("body", {}), context)

        response = self.client.request(
            method=op["method"],
            url=endpoint,
            json=body if body else None,
        )
        response.raise_for_status()

        data = response.json()
        results_path = op["response"]["results_path"]
        results = extract_path(data, results_path, default=[], strict=False)

        if not results:
            return []

        mapping = op["response"]["mapping"]
        search_results = []

        for item in results:
            result_id = extract_path(item, mapping["id"], strict=False)
            result_score = extract_path(item, mapping.get("score", ""), strict=False)
            result_payload = extract_path(
                item, mapping.get("payload", ""), default={}, strict=False
            )

            if result_payload and "_original_id" in result_payload:
                result_id = result_payload["_original_id"]

            search_result = SearchResult(
                id=str(result_id),
                score=float(result_score) if result_score is not None else None,
                payload=result_payload if result_payload else {},
            )
            search_results.append(search_result)

        return search_results

    def upsert(self, collection: str, points: List[Dict]) -> None:
        """Insert or update points in collection."""
        if "upsert" not in self.spec.operations:
            raise NotImplementedError(f"{self.plugin_name} does not support upsert")

        if points and "vector" in points[0]:
            self._vector_dim = len(points[0]["vector"])

        op = self.spec.operations["upsert"]

        converted_points = self._convert_point_ids(points)
        transformed_points = self.spec.transform_points(converted_points, "upsert")

        context = {
            "collection": collection,
            "points": transformed_points,
            "vector_dim": self._vector_dim,
            **self.kwargs,
        }

        endpoint = Template(op["endpoint"]).render(context)
        body = self.spec.render_template(op.get("body", {}), context)

        response = self.client.request(
            method=op["method"],
            url=endpoint,
            json=body,
        )

        if response.status_code == 404 and op.get("auto_create_collection"):
            self._auto_create_collection(collection, op, context)
            response = self.client.request(
                method=op["method"],
                url=endpoint,
                json=body,
            )

        response.raise_for_status()

    def _auto_create_collection(
        self,
        collection: str,
        op: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Auto-create a collection before upsert."""
        create_endpoint = op.get("create_collection_endpoint")
        create_method = op.get("create_collection_method", "PUT")
        create_body_template = op.get("create_collection_body", {})

        if not create_endpoint:
            raise ValueError(
                f"auto_create_collection is true but create_collection_endpoint "
                f"is not specified in {self.plugin_name} YAML"
            )

        endpoint = Template(create_endpoint).render(context)
        body = self.spec.render_template(create_body_template, context)

        response = self.client.request(
            method=create_method,
            url=endpoint,
            json=body,
        )

        if response.status_code not in (200, 201, 409):
            response.raise_for_status()

    def create_collection(self, name: str, vector_size: int) -> None:
        """Create a new collection."""
        if "create_collection" not in self.spec.operations:
            raise NotImplementedError(f"{self.plugin_name} does not support create_collection")

        op = self.spec.operations["create_collection"]
        context = {
            "collection": name,
            "vector_size": vector_size,
            **self.kwargs,
        }

        endpoint = Template(op["endpoint"]).render(context)
        body = self.spec.render_template(op.get("body", {}), context)

        response = self.client.request(
            method=op["method"],
            url=endpoint,
            json=body,
        )
        response.raise_for_status()

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if "delete_collection" not in self.spec.operations:
            raise NotImplementedError(f"{self.plugin_name} does not support delete_collection")

        op = self.spec.operations["delete_collection"]
        context = {"collection": name, **self.kwargs}

        endpoint = Template(op["endpoint"]).render(context)

        response = self.client.request(
            method=op["method"],
            url=endpoint,
        )
        response.raise_for_status()

    def list_collections(self) -> List[str]:
        """List all collections."""
        if "list_collections" not in self.spec.operations:
            raise NotImplementedError(f"{self.plugin_name} does not support list_collections")

        op = self.spec.operations["list_collections"]
        context = {**self.kwargs}

        endpoint = Template(op["endpoint"]).render(context)

        response = self.client.get(endpoint)
        response.raise_for_status()

        data = response.json()
        collections = extract_path(
            data, op["response"]["collections_path"], default=[], strict=False
        )

        if not collections:
            return []

        name_field = op["response"].get("name_field", "name")
        return [c[name_field] if isinstance(c, dict) else c for c in collections]

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        if "get_stats" not in self.spec.operations:
            raise NotImplementedError(f"{self.plugin_name} does not support get_collection_stats")

        op = self.spec.operations["get_stats"]
        context = {"collection": collection, **self.kwargs}

        endpoint = Template(op["endpoint"]).render(context)

        response = self.client.get(endpoint)
        response.raise_for_status()

        data = response.json()
        return extract_path(data, op["response"]["stats_path"], default={}, strict=False)

    def __del__(self):
        """Cleanup HTTP client."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass


def load_vector_db_spec(plugin_name: str) -> VectorDBSpec:
    """Load vector DB specification from YAML file."""
    plugins_dir = Path(__file__).parent / "plugins"
    yaml_path = plugins_dir / f"{plugin_name}.yaml"

    if not yaml_path.exists():
        raise ValueError(f"Vector DB plugin '{plugin_name}' not found. " f"Expected: {yaml_path}")

    return VectorDBSpec(yaml_path)


def _get_auto_detected_kwargs(spec: VectorDBSpec, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-detect connection parameters based on YAML spec.

    Uses the 'features.auto_detect' field to determine which service
    detection function to call from fitz.core.detect.

    This is PROVIDER-AGNOSTIC - the detection logic is driven by YAML.
    """
    result = dict(kwargs)

    auto_detect_service = spec.get_auto_detect_service()
    if not auto_detect_service:
        return result

    # Only auto-detect if host/port not explicitly provided
    if "host" in result and "port" in result:
        return result

    try:
        from fitz.core import detect

        # Map service name to detection function
        detection_functions = {
            "qdrant": detect.get_qdrant_connection,
            "ollama": detect.get_ollama_connection,
        }

        detect_func = detection_functions.get(auto_detect_service)
        if detect_func:
            detected_host, detected_port = detect_func()

            if "host" not in result:
                result["host"] = detected_host
            if "port" not in result:
                result["port"] = detected_port

    except ImportError:
        # fitz.core.detect not available, fall back to YAML defaults
        pass

    return result


def create_vector_db_plugin(plugin_name: str, **kwargs):
    """
    Create a vector DB plugin from YAML specification.

    Connection details are AUTO-DETECTED based on the YAML spec's
    'features.auto_detect' field. This is provider-agnostic.

    For HTTP-based plugins: Returns GenericVectorDBPlugin
    For local plugins: Returns the specific Python implementation
    """
    spec = load_vector_db_spec(plugin_name)

    # Handle local implementations (e.g., FAISS)
    if spec.is_local():
        class_path = spec.get_local_class_path()
        if not class_path:
            raise ValueError(f"Local plugin '{plugin_name}' missing python_class specification")

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        PluginClass = getattr(module, class_name)

        return PluginClass(**kwargs)

    # For HTTP-based plugins, auto-detect connection based on YAML spec
    resolved_kwargs = _get_auto_detected_kwargs(spec, kwargs)

    return GenericVectorDBPlugin(spec, **resolved_kwargs)
