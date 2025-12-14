# tools/contract_map/models.py
"""Extract models and protocols from the codebase."""
from __future__ import annotations

import inspect
from typing import Any, List, Type, get_type_hints

from .common import (
    ContractMap,
    MethodContract,
    ModelContract,
    ModelField,
    ProtocolContract,
    fmt_type,
    is_pydantic_model,
    safe_import,
)


def extract_pydantic_fields(model_cls: Type[Any]) -> List[ModelField]:
    """Extract fields from a Pydantic model."""
    fields: List[ModelField] = []

    if hasattr(model_cls, "model_fields"):  # pydantic v2
        mf: dict[str, Any] = getattr(model_cls, "model_fields")
        for name in sorted(mf.keys()):
            f = mf[name]
            ann = getattr(f, "annotation", None)
            required = bool(getattr(f, "is_required", False))
            default = getattr(f, "default", None)
            fields.append(
                ModelField(
                    name=name,
                    type=fmt_type(ann),
                    required=required,
                    default=None if required else repr(default),
                )
            )
        return fields

    if hasattr(model_cls, "__fields__"):  # pydantic v1
        ff: dict[str, Any] = getattr(model_cls, "__fields__")
        for name in sorted(ff.keys()):
            f = ff[name]
            ann = getattr(f, "type_", None)
            required = bool(getattr(f, "required", False))
            default = getattr(f, "default", None)
            fields.append(
                ModelField(
                    name=name,
                    type=fmt_type(ann),
                    required=required,
                    default=None if required else repr(default),
                )
            )
        return fields

    hints = getattr(model_cls, "__annotations__", {}) or {}
    for name in sorted(hints.keys()):
        fields.append(
            ModelField(name=name, type=fmt_type(hints[name]), required=True, default=None)
        )
    return fields


def looks_like_protocol(obj: Any) -> bool:
    """Check if an object looks like a Protocol."""
    if not isinstance(obj, type):
        return False
    return bool(getattr(obj, "_is_protocol", False))


def extract_protocol_methods(proto_cls: Type[Any]) -> List[MethodContract]:
    """Extract method signatures from a Protocol."""
    methods: List[MethodContract] = []

    for name, member in inspect.getmembers(proto_cls):
        if name.startswith("_"):
            continue
        if not inspect.isfunction(member) and not inspect.ismethod(member):
            continue
        if getattr(member, "__qualname__", "").split(".")[0] != proto_cls.__name__:
            continue

        try:
            sig = str(inspect.signature(member))
        except Exception:
            sig = "(...)"

        returns: str | None = None
        try:
            hints = get_type_hints(member, include_extras=True)
            if "return" in hints:
                returns = fmt_type(hints["return"])
        except Exception:
            returns = None

        methods.append(MethodContract(name=name, signature=sig, returns=returns))

    methods.sort(key=lambda m: m.name)
    return methods


def extract_models(cm: ContractMap, *, verbose: bool) -> None:
    """Extract all models from the codebase."""
    model_modules = [
        "fitz.core.config.schema",
        "fitz.core.models.chunk",
        "fitz.core.models.document",
        "fitz.pipeline.config.schema",
        "fitz.ingest.config.schema",
        "fitz.ingest.ingestion.base",
    ]

    for mod_name in model_modules:
        mod = safe_import(cm, mod_name, verbose=verbose)
        if mod is None:
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue

            if is_pydantic_model(obj):
                cm.models.append(
                    ModelContract(
                        module=mod_name,
                        name=obj.__name__,
                        fields=extract_pydantic_fields(obj),
                    )
                )
                continue

            if obj.__name__ in {"RawDocument"} and hasattr(obj, "__annotations__"):
                hints = obj.__annotations__ or {}
                fields = [
                    ModelField(name=k, type=fmt_type(hints[k]), required=True)
                    for k in sorted(hints.keys())
                ]
                cm.models.append(ModelContract(module=mod_name, name=obj.__name__, fields=fields))

    cm.models.sort(key=lambda m: (m.module, m.name))


def extract_protocols(cm: ContractMap, *, verbose: bool) -> None:
    """Extract all protocols from the codebase."""
    protocol_modules = [
        "fitz.core.llm.chat.base",
        "fitz.core.llm.embedding.base",
        "fitz.core.llm.rerank.base",
        "fitz.core.vector_db.base",
        "fitz.retrieval.base",
        "fitz.pipeline.pipeline.base",
        "fitz.ingest.chunking.base",
        "fitz.ingest.ingestion.base",
    ]

    for mod_name in protocol_modules:
        mod = safe_import(cm, mod_name, verbose=verbose)
        if mod is None:
            continue

        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue
            if looks_like_protocol(obj):
                cm.protocols.append(
                    ProtocolContract(
                        module=mod_name,
                        name=obj.__name__,
                        methods=extract_protocol_methods(obj),
                    )
                )

    cm.protocols.sort(key=lambda p: (p.module, p.name))


def render_models_section(cm: ContractMap) -> str:
    """Render the Models section of the report."""
    lines = ["## Models"]

    for m in cm.models:
        lines.append(f"### `{m.module}.{m.name}`")
        if not m.fields:
            lines.append("- (no fields found)")
            lines.append("")
            continue
        for f in m.fields:
            req = "required" if f.required else "optional"
            default = "" if f.default is None else f" (default={f.default})"
            lines.append(f"- `{f.name}`: `{f.type}` — {req}{default}")
        lines.append("")

    return "\n".join(lines)


def render_protocols_section(cm: ContractMap) -> str:
    """Render the Protocols section of the report."""
    lines = ["## Protocols"]

    for p in cm.protocols:
        lines.append(f"### `{p.module}.{p.name}`")
        if not p.methods:
            lines.append("- (no methods found)")
            lines.append("")
            continue
        for m in p.methods:
            ret = f" -> {m.returns}" if m.returns else ""
            lines.append(f"- `{m.name}{m.signature}`{ret}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from .common import ContractMap

    cm = ContractMap(meta={"test": "models_extraction"})

    print("Extracting models...")
    extract_models(cm, verbose=True)
    print(f"Found {len(cm.models)} models")

    print("\nExtracting protocols...")
    extract_protocols(cm, verbose=True)
    print(f"Found {len(cm.protocols)} protocols")

    print("\n" + "=" * 80)
    print(render_models_section(cm))
    print(render_protocols_section(cm))
