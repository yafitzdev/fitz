# tools/cli_map/__main__.py
"""
CLI Map - extract and display CLI structure from the Typer app.

Usage:
    python -m tools.cli_map            # Full demo with parallel queries (~2 min)
    python -m tools.cli_map --minimal  # Skip code corpus (~1 min)
    python -m tools.cli_map --help-only  # Just --help text for all commands
    python -m tools.cli_map --compact  # One-line summary per command
    python -m tools.cli_map --full     # Full details with types and defaults
    python -m tools.cli_map --json     # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ensure repo root is in path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


@dataclass
class ParamInfo:
    """Information about a command parameter (argument or option)."""

    name: str
    kind: str  # "argument" or "option"
    type_hint: str
    required: bool
    default: Optional[str]
    help: str
    flags: list[str] = field(default_factory=list)  # e.g., ["--verbose", "-v"]


@dataclass
class CommandInfo:
    """Information about a CLI command."""

    name: str
    help: str
    arguments: list[ParamInfo] = field(default_factory=list)
    options: list[ParamInfo] = field(default_factory=list)


@dataclass
class CLIMap:
    """Complete CLI structure."""

    app_name: str
    app_help: str
    commands: list[CommandInfo] = field(default_factory=list)
    subapps: dict[str, "CLIMap"] = field(default_factory=dict)


def extract_param_info(param: Any) -> ParamInfo:
    """Extract parameter info from a Click parameter."""
    import click

    # Determine kind
    if isinstance(param, click.Argument):
        kind = "argument"
        flags = []
    else:
        kind = "option"
        flags = list(param.opts) if hasattr(param, "opts") else []

    # Get type hint
    type_hint = "str"
    if param.type:
        type_name = getattr(param.type, "name", None)
        if type_name:
            type_hint = type_name
        elif hasattr(param.type, "__class__"):
            type_hint = param.type.__class__.__name__

    # Get default
    default = None
    if param.default is not None and param.default != ():
        default = repr(param.default)

    # Get help
    help_text = getattr(param, "help", "") or ""

    return ParamInfo(
        name=param.name,
        kind=kind,
        type_hint=type_hint,
        required=param.required,
        default=default,
        help=help_text,
        flags=flags,
    )


def extract_command_info(name: str, cmd: Any) -> CommandInfo:
    """Extract command info from a Click/Typer command."""
    help_text = cmd.help or cmd.callback.__doc__ or ""
    help_text = help_text.strip().split("\n")[0]  # First line only

    arguments = []
    options = []

    for param in cmd.params:
        info = extract_param_info(param)
        if info.kind == "argument":
            arguments.append(info)
        else:
            # Skip the --help option
            if info.name != "help":
                options.append(info)

    return CommandInfo(
        name=name,
        help=help_text,
        arguments=arguments,
        options=options,
    )


def extract_cli_map(app: Any) -> CLIMap:
    """Extract the full CLI map from a Typer app."""
    import click
    import typer

    # Get underlying click group from Typer
    click_app = typer.main.get_command(app)

    app_name = getattr(click_app, "name", "fitz") or "fitz"
    app_help = getattr(click_app, "help", "") or ""

    cli_map = CLIMap(app_name=app_name, app_help=app_help)

    # Extract commands
    if hasattr(click_app, "commands"):
        for cmd_name, cmd in sorted(click_app.commands.items()):
            if isinstance(cmd, click.Group):
                # Subapp (e.g., benchmark) - recurse
                sub_map = CLIMap(app_name=cmd_name, app_help=cmd.help or "")
                if hasattr(cmd, "commands"):
                    for sub_name, sub_cmd in sorted(cmd.commands.items()):
                        sub_map.commands.append(extract_command_info(sub_name, sub_cmd))
                cli_map.subapps[cmd_name] = sub_map
            else:
                cli_map.commands.append(extract_command_info(cmd_name, cmd))

    return cli_map


def render_compact(cli_map: CLIMap) -> str:
    """Render CLI map in compact format - one line per command."""
    lines = []

    # Stats
    total_commands = len(cli_map.commands)
    total_args = sum(len(c.arguments) for c in cli_map.commands)
    total_opts = sum(len(c.options) for c in cli_map.commands)
    total_subcommands = sum(len(s.commands) for s in cli_map.subapps.values())

    # Header
    lines.append("=" * 72)
    lines.append(f"  CLI MAP: {cli_map.app_name}")
    lines.append(
        f"  {total_commands} commands | {total_args} args | {total_opts} options | {len(cli_map.subapps)} subapps ({total_subcommands} subcommands)"
    )
    lines.append("=" * 72)
    lines.append("")

    # Commands
    for cmd in cli_map.commands:
        # Build signature
        sig_parts = [f"fitz {cmd.name}"]

        for arg in cmd.arguments:
            if arg.required:
                sig_parts.append(f"<{arg.name}>")
            else:
                sig_parts.append(f"[{arg.name}]")

        for opt in cmd.options:
            flag = opt.flags[0] if opt.flags else f"--{opt.name}"
            if opt.type_hint == "BOOL" or opt.type_hint == "boolean":
                sig_parts.append(f"[{flag}]")
            elif opt.required:
                sig_parts.append(f"{flag}=<{opt.type_hint}>")
            else:
                sig_parts.append(f"[{flag}]")

        sig = " ".join(sig_parts)
        lines.append(f"  {sig}")
        lines.append(f"    └─ {cmd.help}")
        lines.append("")

    # Subapps
    for subapp_name, subapp in sorted(cli_map.subapps.items()):
        lines.append(f"  fitz {subapp_name} ...")
        lines.append(f"    └─ {subapp.app_help or '(subcommand group)'}")
        for cmd in subapp.commands:
            sig_parts = [f"      fitz {subapp_name} {cmd.name}"]
            for arg in cmd.arguments:
                sig_parts.append(f"[{arg.name}]" if not arg.required else f"<{arg.name}>")
            sig = " ".join(sig_parts)
            lines.append(sig)
        lines.append("")

    lines.append("-" * 72)

    return "\n".join(lines)


def render_full(cli_map: CLIMap) -> str:
    """Render CLI map with full details - types, defaults, help text."""
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"CLI MAP: {cli_map.app_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(cli_map.app_help)
    lines.append("")
    lines.append("-" * 80)

    # Commands
    for cmd in cli_map.commands:
        lines.append("")
        lines.append(f"┌── fitz {cmd.name}")
        lines.append(f"│   {cmd.help}")
        lines.append("│")

        if cmd.arguments:
            lines.append("│   Arguments:")
            for arg in cmd.arguments:
                req = "required" if arg.required else "optional"
                default = f" = {arg.default}" if arg.default else ""
                lines.append(f"│     {arg.name} : {arg.type_hint} ({req}{default})")
                if arg.help:
                    lines.append(f"│       └─ {arg.help}")

        if cmd.options:
            lines.append("│   Options:")
            for opt in cmd.options:
                flags = ", ".join(opt.flags) if opt.flags else f"--{opt.name}"
                req = "required" if opt.required else "optional"
                default = f" = {opt.default}" if opt.default else ""
                lines.append(f"│     {flags} : {opt.type_hint} ({req}{default})")
                if opt.help:
                    lines.append(f"│       └─ {opt.help}")

        lines.append("└" + "─" * 40)

    # Subapps
    for subapp_name, subapp in sorted(cli_map.subapps.items()):
        lines.append("")
        lines.append(f"┌── fitz {subapp_name} (subcommand group)")
        lines.append(f"│   {subapp.app_help}")
        lines.append("│")

        for cmd in subapp.commands:
            lines.append(f"│   ├── {cmd.name}")
            lines.append(f"│   │   {cmd.help}")

            if cmd.arguments:
                for arg in cmd.arguments:
                    req = "required" if arg.required else "optional"
                    lines.append(f"│   │     {arg.name} : {arg.type_hint} ({req})")

            if cmd.options:
                for opt in cmd.options:
                    flags = ", ".join(opt.flags) if opt.flags else f"--{opt.name}"
                    lines.append(f"│   │     {flags} : {opt.type_hint}")

        lines.append("└" + "─" * 40)

    return "\n".join(lines)


def render_json(cli_map: CLIMap) -> str:
    """Render CLI map as JSON."""

    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            result = {}
            for k, v in asdict(obj).items():
                result[k] = v
            return result
        return obj

    return json.dumps(asdict(cli_map), indent=2)


def render_live_output(minimal: bool = False) -> str:
    """
    Run systematic interactive CLI tests following a 12-step test plan.

    Tests all interactive CLI workflows without using CLI flags.
    Focus on testing fitz-ai features: epistemic honesty, citations,
    hierarchical RAG, multi-source synthesis, and grounding.

    Args:
        minimal: If True, skip code corpus (steps 10-12)
    """
    import os
    import shutil
    import subprocess
    import tempfile

    lines = []

    # Find fitz executable
    fitz_exe = shutil.which("fitz")
    if not fitz_exe:
        venv_dir = Path(sys.executable).parent
        fitz_exe = venv_dir / "fitz.exe"
        if not fitz_exe.exists():
            fitz_exe = venv_dir / "fitz"
        fitz_exe = str(fitz_exe)

    def run_cmd(*args: str, timeout: int = 60, stdin_input: str = None) -> str:
        """Run fitz command and capture output.

        Args:
            *args: Command arguments (positional only, no flags allowed in test plan)
            timeout: Timeout in seconds
            stdin_input: Input to send to stdin (simulates user typing + Enter)
        """
        try:
            env = {**os.environ, "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [fitz_exe, *args],
                capture_output=True,
                timeout=timeout,
                env=env,
                encoding="utf-8",
                errors="replace",
                input=stdin_input,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            output = stdout + stderr
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "(timed out)"
        except Exception as e:
            return f"ERROR: {e}"

    def section(step: str, title: str, output: str) -> None:
        """Add a numbered section to the output."""
        lines.append("=" * 72)
        lines.append(f"  STEP {step}: {title}")
        lines.append("=" * 72)
        lines.append(output)
        lines.append("")

    def subsection(title: str, output: str) -> None:
        """Add a subsection to the output."""
        lines.append("-" * 72)
        lines.append(f"  {title}")
        lines.append("-" * 72)
        lines.append(output)
        lines.append("")

    # Create temp directory with test documents
    temp_dir = tempfile.mkdtemp(prefix="fitz_cli_map_")
    docs_collection = "cli_map_docs"
    code_collection = "cli_map_code"

    try:
        # =====================================================================
        # Header
        # =====================================================================
        lines.append("=" * 72)
        lines.append("  FITZ CLI MAP - Systematic Interactive Test")
        lines.append("=" * 72)
        lines.append("")
        lines.append("This test validates all interactive CLI workflows.")
        lines.append("No CLI flags are used - only interactive prompts.")
        lines.append("")
        lines.append("Test Plan:")
        lines.append("  1) fitz engine - select fitz_rag")
        lines.append("  2) fitz init - set cohere, faiss as defaults")
        lines.append("  3) fitz doctor, fitz config")
        lines.append("  4) fitz quickstart - test one query")
        lines.append("  5) fitz query - query quickstart collection")
        lines.append("  6) fitz collections - delete quickstart")
        lines.append("  7) fitz ingest - ingest docs (1 .md, 1 .txt, 1 .pdf)")
        lines.append("  8) fitz chat - 5 questions testing features")
        lines.append("  9) fitz collections - delete docs collection")
        lines.append("  10) fitz ingest - ingest contract_map codebase")
        lines.append("  11) fitz query - 3 questions for codebase")
        lines.append("  12) fitz collections - delete code collection")
        lines.append("")

        # =====================================================================
        # STEP 1: fitz engine - select fitz_rag
        # =====================================================================
        # Interactive: Enter confirms current default (fitz_rag)
        section(
            "1",
            "fitz engine - select fitz_rag",
            run_cmd("engine", stdin_input="\n"),
        )

        # =====================================================================
        # STEP 2: fitz init - set cohere, faiss as defaults
        # =====================================================================
        # Interactive init wizard flow for fitz_rag:
        # 1. Engine selection: Enter (confirm fitz_rag)
        # 2. Chat plugin: Enter (cohere default)
        # 3. Smart model: Enter (accept default)
        # 4. Fast model: Enter (accept default)
        # 5. Embedding plugin: Enter (cohere default)
        # 6. Embedding model: Enter (accept default)
        # 7. Rerank plugin: Enter (cohere default)
        # 8. Rerank model: Enter (accept default)
        # 9. Vector DB: Enter (local_faiss is default)
        # 10. Retrieval: (may be skipped if only one option)
        # 11. Chunker: Enter (default)
        # 12. Chunk size: Enter (default)
        # 13. Chunk overlap: Enter (default)
        # 14. Confirm overwrite: y
        # Note: Using fewer \n's since Retrieval prompt may be skipped
        init_stdin = "\n\n\n\n\n\n\n\n\n\n\n\ny\n"
        section(
            "2",
            "fitz init - set cohere, faiss defaults",
            run_cmd("init", stdin_input=init_stdin, timeout=60),
        )

        # =====================================================================
        # STEP 3: fitz doctor, fitz config
        # =====================================================================
        doctor_output = run_cmd("doctor", timeout=30)
        config_output = run_cmd("config", timeout=10)

        section("3a", "fitz doctor", doctor_output)
        subsection("fitz config", config_output)

        # =====================================================================
        # Create test documents for quickstart
        # =====================================================================
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()

        # Create diverse test documents
        (docs_dir / "about.md").write_text(
            "# About Fitz AI\n\n"
            "Fitz is a local-first RAG framework for building knowledge engines.\n\n"
            "## Core Features\n"
            "- **Zero-config quickstart**: Get started with one command\n"
            "- **Epistemic honesty**: Says 'I don't know' when evidence is insufficient\n"
            "- **Full provenance tracking**: Every answer includes citations\n"
            "- **Hierarchical RAG**: Multi-level summaries for better context\n\n"
            "## Architecture\n"
            "Fitz uses a modular plugin architecture:\n"
            "- Chat plugins (Cohere, OpenAI, Ollama)\n"
            "- Embedding plugins (Cohere, OpenAI)\n"
            "- Vector DB plugins (FAISS, Qdrant)\n"
            "- Reranking plugins (Cohere)\n",
            encoding="utf-8",
        )

        (docs_dir / "installation.txt").write_text(
            "Installation Instructions\n"
            "=========================\n\n"
            "To install Fitz, run:\n"
            "  pip install fitz-ai\n\n"
            "Requirements:\n"
            "- Python 3.10 or higher\n"
            "- 8GB RAM recommended\n"
            "- API key for your chosen provider (Cohere, OpenAI, etc.)\n\n"
            "Quick Start:\n"
            "  fitz quickstart ./docs 'What is this about?'\n\n"
            "This will ingest your documents and answer your question in one command.\n",
            encoding="utf-8",
        )

        # Create a simple text file that could represent PDF content
        # (Real PDF testing would require a PDF library)
        (docs_dir / "changelog.txt").write_text(
            "Changelog\n"
            "=========\n\n"
            "Version 0.4.0 (2024-01-15)\n"
            "--------------------------\n"
            "- Added hierarchical summaries (L0/L1/L2 levels)\n"
            "- Improved citation accuracy to 95%+\n"
            "- New epistemic constraints for honesty\n\n"
            "Version 0.3.0 (2024-01-01)\n"
            "--------------------------\n"
            "- Initial public release\n"
            "- Support for Cohere, OpenAI providers\n"
            "- FAISS and Qdrant vector databases\n",
            encoding="utf-8",
        )

        # =====================================================================
        # STEP 4: fitz quickstart - test one query
        # =====================================================================
        # quickstart takes positional args: SOURCE QUESTION
        # Interactive: may prompt for collection name → use default with Enter
        section(
            "4",
            f"fitz quickstart {docs_dir} 'What is Fitz?'",
            run_cmd(
                "quickstart",
                str(docs_dir),
                "What is Fitz?",
                stdin_input="\n",
                timeout=180,
            ),
        )

        # =====================================================================
        # STEP 5: fitz query - query quickstart collection
        # =====================================================================
        # Interactive query on quickstart collection
        # Flow: question first, then collection selection (Enter for default)
        from fitz_ai.cli.context import CLIContext

        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        collections = client.list_collections()

        if "quickstart" in collections:
            # Interactive: enter question first, then Enter for default collection
            # (quickstart should be default after just being created)
            section(
                "5",
                "fitz query - query quickstart collection",
                run_cmd(
                    "query",
                    stdin_input="How do I install Fitz?\n\n",
                    timeout=120,
                ),
            )

        # =====================================================================
        # STEP 6: fitz collections - delete quickstart
        # =====================================================================
        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        collections = sorted(client.list_collections())

        if "quickstart" in collections:
            qs_index = collections.index("quickstart") + 1
            # Interactive: Enter for vector_db, select collection, Delete (2), confirm (y), Exit (4)
            section(
                "6",
                "fitz collections - delete quickstart",
                run_cmd("collections", stdin_input=f"\n{qs_index}\n2\ny\n4\n"),
            )

        # =====================================================================
        # STEP 7: fitz ingest - ingest docs freshly
        # =====================================================================
        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        existing_collections = client.list_collections()

        # Interactive: collection menu (0 for new if exists) + name
        if existing_collections:
            ingest_stdin = f"0\n{docs_collection}\n"
        else:
            ingest_stdin = f"{docs_collection}\n"

        section(
            "7",
            f"fitz ingest {docs_dir} - create {docs_collection}",
            run_cmd("ingest", str(docs_dir), stdin_input=ingest_stdin, timeout=180),
        )

        # =====================================================================
        # STEP 8: fitz chat - 5 questions testing features
        # =====================================================================
        # Test features:
        # 1. Basic retrieval + citations
        # 2. Epistemic honesty (ask something not in docs)
        # 3. Multi-source synthesis
        # 4. Specific fact retrieval
        # 5. Summary/overview question

        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        collections = sorted(client.list_collections())

        if docs_collection in collections:
            # Chat questions designed to test fitz-ai features
            chat_questions = [
                "What are the core features of Fitz?",  # Basic retrieval
                "What is the weather in Tokyo?",  # Epistemic honesty (not in docs)
                "Summarize the installation process and requirements",  # Multi-source
                "What version added hierarchical summaries?",  # Specific fact
                "Give me an overview of what changed in recent versions",  # Summary
            ]

            # Build stdin: select collection (Enter for default), then questions, then exit
            chat_stdin = "\n"  # Accept default collection
            for q in chat_questions:
                chat_stdin += f"{q}\n"
            chat_stdin += "exit\n"

            section(
                "8",
                "fitz chat - 5 questions testing features",
                run_cmd("chat", stdin_input=chat_stdin, timeout=300),
            )

        # =====================================================================
        # STEP 9: fitz collections - delete docs collection
        # =====================================================================
        ctx = CLIContext.load()
        client = ctx.get_vector_db_client()
        collections = sorted(client.list_collections())

        if docs_collection in collections:
            coll_index = collections.index(docs_collection) + 1
            section(
                "9",
                f"fitz collections - delete {docs_collection}",
                run_cmd("collections", stdin_input=f"\n{coll_index}\n2\ny\n4\n"),
            )

        # =====================================================================
        # STEPS 10-12: Code corpus (contract_map)
        # =====================================================================
        if not minimal:
            contract_map_dir = Path(__file__).parent.parent / "contract_map"

            lines.append("")
            lines.append("=" * 72)
            lines.append("  CODE CORPUS TEST (contract_map)")
            lines.append("=" * 72)
            lines.append("")

            # STEP 10: fitz ingest - ingest contract_map codebase
            ctx = CLIContext.load()
            client = ctx.get_vector_db_client()
            existing_collections = client.list_collections()

            if existing_collections:
                ingest_stdin = f"0\n{code_collection}\n"
            else:
                ingest_stdin = f"{code_collection}\n"

            section(
                "10",
                f"fitz ingest {contract_map_dir} - create {code_collection}",
                run_cmd(
                    "ingest", str(contract_map_dir), stdin_input=ingest_stdin, timeout=180
                ),
            )

            # STEP 11: fitz query - 3 questions for codebase
            ctx = CLIContext.load()
            client = ctx.get_vector_db_client()
            collections = sorted(client.list_collections())

            if code_collection in collections:
                code_index = collections.index(code_collection) + 1

                code_questions = [
                    ("What does the contract_map tool do?", "Purpose query"),
                    ("How are import violations detected?", "Implementation detail"),
                    ("What modules are in the codebase?", "Structure query"),
                ]

                # Query flow: question first, then collection selection
                for question, desc in code_questions:
                    subsection(
                        f"fitz query: {desc}",
                        run_cmd(
                            "query",
                            stdin_input=f"{question}\n{code_index}\n",
                            timeout=120,
                        ),
                    )

            # STEP 12: fitz collections - delete code collection
            ctx = CLIContext.load()
            client = ctx.get_vector_db_client()
            collections = sorted(client.list_collections())

            if code_collection in collections:
                code_index = collections.index(code_collection) + 1
                section(
                    "12",
                    f"fitz collections - delete {code_collection}",
                    run_cmd("collections", stdin_input=f"\n{code_index}\n2\ny\n4\n"),
                )

    finally:
        # =====================================================================
        # Cleanup
        # =====================================================================
        lines.append("=" * 72)
        lines.append("  CLEANUP")
        lines.append("=" * 72)

        # Delete temp directory
        try:
            shutil.rmtree(temp_dir)
            lines.append(f"Deleted temp dir: {temp_dir}")
        except Exception as e:
            lines.append(f"Could not delete temp dir: {e}")

        lines.append("")
        lines.append("=" * 72)
        lines.append("  TEST COMPLETE")
        lines.append("=" * 72)

    return "\n".join(lines)


def render_help_output(cli_map: CLIMap) -> str:
    """Render actual --help output for all commands."""
    import shutil
    import subprocess

    lines = []

    # Find fitz executable
    fitz_exe = shutil.which("fitz")
    if not fitz_exe:
        # Try in the same venv as current python
        venv_dir = Path(sys.executable).parent
        fitz_exe = venv_dir / "fitz"
        if not fitz_exe.exists():
            fitz_exe = venv_dir / "fitz.exe"
        fitz_exe = str(fitz_exe)

    def run_help(*args: str) -> str:
        """Run fitz with args and capture output."""
        import os

        try:
            # Disable rich formatting for cleaner output
            env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(
                [fitz_exe, *args, "--help"],
                capture_output=True,
                timeout=10,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            return stdout.strip() or stderr.strip()
        except Exception as e:
            return f"ERROR: {e}"

    # Main app help
    lines.append("=" * 72)
    lines.append("  fitz --help")
    lines.append("=" * 72)
    lines.append(run_help())
    lines.append("")

    # Each command's help
    for cmd in cli_map.commands:
        lines.append("=" * 72)
        lines.append(f"  fitz {cmd.name} --help")
        lines.append("=" * 72)
        lines.append(run_help(cmd.name))
        lines.append("")

    # Subapps
    for subapp_name, subapp in sorted(cli_map.subapps.items()):
        lines.append("=" * 72)
        lines.append(f"  fitz {subapp_name} --help")
        lines.append("=" * 72)
        lines.append(run_help(subapp_name))
        lines.append("")

        # Subcommands
        for cmd in subapp.commands:
            lines.append("-" * 72)
            lines.append(f"  fitz {subapp_name} {cmd.name} --help")
            lines.append("-" * 72)
            lines.append(run_help(subapp_name, cmd.name))
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Display CLI structure at a glance.")
    parser.add_argument(
        "--help-only",
        action="store_true",
        help="Show only --help text (no live execution)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Show compact overview (one line per command)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full details (types, defaults, help)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Minimal mode: quickstart + docs only, skip code corpus (~1 min)",
    )

    args = parser.parse_args(argv)

    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # Import the CLI app
    try:
        from fitz_ai.cli import app
    except ImportError as e:
        print(f"Error: Could not import CLI app: {e}", file=sys.stderr)
        return 1

    # Extract CLI structure
    cli_map = extract_cli_map(app)

    # Render output (live execution is default)
    if args.json:
        print(render_json(cli_map))
    elif args.compact:
        print(render_compact(cli_map))
    elif args.full:
        print(render_full(cli_map))
    elif args.help_only:
        print(render_help_output(cli_map))
    else:
        # Default: run actual commands with test data (queries run in parallel)
        print(render_live_output(minimal=args.minimal))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
