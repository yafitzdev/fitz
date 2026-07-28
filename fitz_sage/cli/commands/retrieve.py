# fitz_sage/cli/commands/retrieve.py
"""Retrieve governed evidence without answer synthesis."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from fitz_sage.cli.ui import display_evidence_pack, ui
from fitz_sage.core import Query
from fitz_sage.logging.logger import get_logger
from fitz_sage.runtime import create_engine, get_default_engine, get_engine_registry

logger = get_logger(__name__)

_DAEMON_SPAWNED = "spawned"
_DAEMON_RUNNING = "running"
_DAEMON_FAILED = "failed"


def _enrichment_needs_daemon(status: dict) -> bool:
    """Return whether a detached worker should continue enrichment."""
    enrichment = status.get("enrichment") if isinstance(status, dict) else None
    if not isinstance(enrichment, dict):
        return False
    return (
        bool(enrichment.get("pending", 0))
        or bool(enrichment.get("failed", 0))
        or enrichment.get("finalization") in {"pending", "failed"}
    )


def _daemon_pid_path(collection: str, cwd: Path) -> Path:
    """Return the PID file path for a collection's enrichment worker."""
    return cwd / ".fitz" / "collections" / collection / "enrichment_daemon.pid"


def _daemon_log_path(collection: str, cwd: Path) -> Path:
    """Return the detached enrichment worker log path."""
    return cwd / ".fitz" / "collections" / collection / "enrichment_daemon.log"


def _pid_is_running(pid: int) -> bool:
    """Return whether a process id appears to still be alive."""
    if os.name == "nt":
        return _windows_pid_is_running(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _windows_pid_is_running(pid: int) -> bool:
    """Return whether a Windows process id is active."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_query_limited_information = 0x1000
    still_active = 259

    handle = kernel32.OpenProcess(process_query_limited_information, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _read_running_daemon_pid(pid_path: Path) -> int | None:
    """Return an active daemon PID from a PID file, if one exists."""
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_is_running(pid) else None


def _spawn_enrichment_daemon(collection: str, engine_name: str, cwd: Path) -> str:
    """Start a detached process that enriches this collection."""
    pid_path = _daemon_pid_path(collection, cwd)
    if _read_running_daemon_pid(pid_path) is not None:
        return _DAEMON_RUNNING

    cmd = [
        sys.executable,
        "-m",
        "fitz_sage.cli.cli",
        "enrichment-daemon",
        "--collection",
        collection,
        "--engine",
        engine_name,
    ]

    log_handle = None
    log_path = _daemon_log_path(collection, cwd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = log_path.open("ab")
    except Exception as e:
        logger.debug(f"Failed to open enrichment daemon log: {e}")
        return _DAEMON_FAILED

    kwargs: dict = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": log_handle,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
    else:
        kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(cmd, **kwargs)
        pid_path.write_text(str(process.pid), encoding="utf-8")
        return _DAEMON_SPAWNED
    except Exception as e:
        logger.debug(f"Failed to spawn enrichment daemon: {e}")
        return _DAEMON_FAILED
    finally:
        if log_handle is not None:
            log_handle.close()


def _default_source(source: Optional[Path], collection: Optional[str]) -> Optional[Path]:
    """Use the current directory when the user did not choose a collection."""
    if source is not None:
        return source
    if collection is None:
        return Path.cwd()
    return None


def _collection_name_for_source(source: Path) -> str:
    """Derive the default collection name from the selected source directory."""
    from fitz_sage.core.collections import collection_name_from_path

    return collection_name_from_path(source)


def _persisted_source_matches(collection: str, source: Path, cwd: Path) -> bool:
    """Return whether a collection already points at this source directory."""
    collection_dir = cwd / ".fitz" / "collections" / collection
    source_dir_path = collection_dir / "source_dir.txt"
    manifest_path = collection_dir / "manifest.json"
    if not source_dir_path.exists() or not manifest_path.exists():
        return False

    try:
        persisted = Path(source_dir_path.read_text(encoding="utf-8").strip()).resolve()
    except OSError:
        return False

    selected = source.resolve()
    selected_dir = selected.parent if selected.is_file() else selected
    return persisted == selected_dir.resolve()


def command(
    question: Optional[str],
    source: Optional[Path],
    collection: Optional[str],
    engine: Optional[str],
    *,
    output_format: str = "text",
    top_k: int | None = None,
    trace_path: Path | None = None,
    trace_content: bool = False,
) -> None:
    """Run retrieval-first evidence mode."""
    output_format = output_format.lower().strip()
    if output_format not in {"text", "json"}:
        ui.error("--format must be 'text' or 'json'.")
        raise typer.Exit(1)
    if top_k is not None and top_k < 1:
        ui.error("--top-k must be greater than zero.")
        raise typer.Exit(1)
    if trace_content and trace_path is None:
        ui.error("--trace-content requires --trace PATH.")
        raise typer.Exit(1)
    effective_source = _default_source(source, collection)
    if effective_source is not None and not effective_source.exists():
        ui.error(f"Path does not exist: {effective_source}")
        raise typer.Exit(1)

    registry = get_engine_registry()
    engine_name = engine or get_default_engine()
    if engine_name not in registry.list():
        ui.error(f"Unknown engine: '{engine_name}'. Available: {', '.join(registry.list())}")
        raise typer.Exit(1)

    caps = registry.get_capabilities(engine_name)
    if not caps.supports_persistent_ingest:
        ui.error(f"Engine '{engine_name}' does not support retrieval evidence mode.")
        raise typer.Exit(1)

    selected_collection = _select_collection(registry, engine_name, collection, effective_source)
    question_text = question if question is not None else ui.prompt_text("Question")
    metadata = {"top_k": top_k} if top_k is not None else {}
    progress = None if output_format == "json" else ui.info
    cwd = Path.cwd()
    source_was_explicit = source is not None

    try:
        engine_instance = create_engine(engine_name)
        if effective_source is not None:
            if not source_was_explicit and _persisted_source_matches(
                selected_collection, effective_source, cwd
            ):
                if progress:
                    progress(f"Loading collection '{selected_collection}'...")
                engine_instance.load(selected_collection)
            else:
                if progress:
                    progress(f"Registering {effective_source}...")
                engine_instance.point(
                    effective_source,
                    selected_collection,
                    start_worker=False,
                    progress=progress,
                )
        else:
            if progress:
                progress(f"Loading collection '{selected_collection}'...")
            engine_instance.load(selected_collection)

        query = Query(text=question_text, metadata=metadata)
        if trace_path is not None:
            run = engine_instance.trace(
                query,
                progress=progress,
                top_k=top_k,
            )
            pack = run.evidence
            written_trace = run.write(trace_path, include_content=trace_content)
            if output_format != "json":
                ui.info(f"Trace written to {written_trace}.")
        else:
            pack = engine_instance.evidence(
                query,
                progress=progress,
                top_k=top_k,
            )
        if output_format == "json":
            print(pack.to_json())
        else:
            display_evidence_pack(pack, max_items=top_k or 10)
        if _enrichment_needs_daemon(pack.indexing_status):
            stop_worker = getattr(engine_instance, "stop_background_enrichment", None)
            if callable(stop_worker):
                stop_worker()
            if output_format != "json":
                daemon_status = _spawn_enrichment_daemon(selected_collection, engine_name, cwd)
                if daemon_status == _DAEMON_SPAWNED:
                    ui.info("Enrichment continues in the background.")
                elif daemon_status == _DAEMON_RUNNING:
                    ui.info("Enrichment daemon already running.")
            elif output_format == "json":
                _spawn_enrichment_daemon(selected_collection, engine_name, cwd)
    except Exception as e:
        ui.error(f"Retrieve failed: {e}")
        logger.debug("Retrieve error", exc_info=True)
        raise typer.Exit(1)


def _select_collection(
    registry,
    engine_name: str,
    requested: Optional[str],
    source: Optional[Path],
) -> str:
    """Resolve target collection for source or collection retrieval."""
    if source is not None:
        return requested or _collection_name_for_source(source)

    collections = [str(name) for name in registry.get_list_collections(engine_name)]
    if not collections:
        ui.warning("No collections found.")
        ui.info("Run 'fitz retrieve \"question\"' from your documents folder to get started.")
        raise typer.Exit(0)

    if requested is not None:
        if requested not in collections:
            ui.error(f"Collection '{requested}' not found. Available: {', '.join(collections)}")
            raise typer.Exit(1)
        return requested

    if len(collections) == 1:
        return str(collections[0])

    return str(ui.prompt_numbered_choice("Collection", collections, collections[0]))
