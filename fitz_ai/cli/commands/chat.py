# fitz_ai/cli/commands/chat.py
"""
Chat command - Conversational RAG interface.

Usage:
    fitz chat                     # Interactive mode (prompts for collection)
    fitz chat -c my_collection    # Specify collection directly
    fitz chat --engine clara      # Use different engine (not yet supported)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from fitz_ai.cli.ui import RICH, console, display_sources, ui
from fitz_ai.cli.utils import get_collections, load_classic_rag_config
from fitz_ai.core.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.runtime import get_default_engine, get_engine_registry, list_engines

logger = get_logger(__name__)


# =============================================================================
# Message Building
# =============================================================================


MAX_HISTORY_MESSAGES = 15


def _build_messages(
    history: List[Dict[str, str]],
    chunks: List[Chunk],
    current_question: str,
) -> List[Dict[str, Any]]:
    """
    Build messages for the LLM with context and history.

    Args:
        history: Previous conversation exchanges (will be trimmed to last 15)
        chunks: Retrieved chunks for current question
        current_question: The current user question

    Returns:
        List of messages ready for the chat client
    """
    # Trim history to last N messages
    recent_history = history[-MAX_HISTORY_MESSAGES:] if history else []

    # Build chunk context
    chunk_context = "\n\n".join(f"[{i}] {chunk.content}" for i, chunk in enumerate(chunks, 1))

    system_prompt = f"""You are a helpful assistant answering questions about a knowledge base.

Use the following retrieved context to answer questions. If the context doesn't contain relevant information, say so honestly.

Retrieved Context:
{chunk_context}

Guidelines:
- Ground your answers in the provided context
- Reference source numbers [1], [2], etc. when citing information
- Be conversational but accurate
- If you don't have enough information, say so"""

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Add conversation history (limited to last N messages)
    messages.extend(recent_history)

    # Add current question
    messages.append({"role": "user", "content": current_question})

    return messages


# =============================================================================
# Chat Display
# =============================================================================

ASSISTANT_INDENT = "    "  # 4 spaces


def _display_user_message(text: str) -> None:
    """Display user message in a panel."""
    if RICH:
        from rich.panel import Panel

        console.print(
            Panel(
                text,
                title="[bold green]You[/bold green]",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )
        )
    else:
        print(f"\nYou: {text}")


def _display_assistant_message(text: str) -> None:
    """Display assistant message in an indented panel."""
    if RICH:
        from rich.markdown import Markdown
        from rich.padding import Padding
        from rich.panel import Panel

        console.print()  # Empty line before bubble
        panel = Panel(
            Markdown(text),
            title="[bold cyan]Assistant[/bold cyan]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(Padding(panel, (0, 0, 0, 12)))  # left padding of 12
    else:
        print(f"\n\nAssistant: {text}")


def _display_welcome(collection: str, engine: str) -> None:
    """Display welcome message."""
    if RICH:
        console.print(
            f"\n[bold green]Chat started[/bold green] with collection: [cyan]{collection}[/cyan] "
            f"(engine: [yellow]{engine}[/yellow])"
        )
        console.print(
            "[dim]Type 'exit' or 'quit' to end the conversation. Press Ctrl+C to interrupt.[/dim]\n"
        )
    else:
        print(f"\nChat started with collection: {collection} (engine: {engine})")
        print("Type 'exit' or 'quit' to end the conversation. Press Ctrl+C to interrupt.\n")


def _display_goodbye() -> None:
    """Display goodbye message."""
    if RICH:
        console.print("\n[dim]Chat ended. Goodbye![/dim]")
    else:
        print("\nChat ended. Goodbye!")


# =============================================================================
# Main Command
# =============================================================================


def command(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to chat with (will prompt if not specified).",
    ),
    engine: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Engine to use. Will prompt if not specified.",
    ),
) -> None:
    """
    Interactive chat with your knowledge base.

    Start a conversation where each question retrieves relevant context
    from your ingested documents. The conversation history is maintained
    throughout the session for natural follow-up questions.

    Examples:
        fitz chat                     # Interactive mode
        fitz chat -c my_collection    # Specify collection
        fitz chat --engine clara      # Use different engine (not yet supported)
    """
    # =========================================================================
    # Engine selection
    # =========================================================================

    available_engines = list_engines()
    registry = get_engine_registry()

    if engine is None:
        # Prompt for engine selection
        ui.header("Fitz Chat", "Conversational RAG")
        print()

        # Build choices with descriptions
        engine_info = registry.list_with_descriptions()
        choices = []
        for name in available_engines:
            desc = engine_info.get(name, "")
            if len(desc) > 60:
                desc = desc[:57] + "..."
            choices.append(f"{name} - {desc}" if desc else name)

        # Get default engine from config (prompt_numbered_choice puts default at position 1)
        default_engine_name = get_default_engine()
        default_choice = next((c for c in choices if c.startswith(default_engine_name)), choices[0])
        selected = ui.prompt_numbered_choice("Engine", choices, default_choice)
        engine = selected.split(" - ")[0]
        print()
    elif engine not in available_engines:
        ui.error(f"Unknown engine: '{engine}'. Available: {', '.join(available_engines)}")
        raise typer.Exit(1)

    # =========================================================================
    # Header (with engine)
    # =========================================================================

    ui.header("Fitz Chat", f"Conversational RAG (engine: {engine})")

    # =========================================================================
    # Capabilities-based routing
    # =========================================================================

    caps = registry.get_capabilities(engine)

    # Check if engine supports chat
    if not caps.supports_chat:
        ui.warning(f"Engine '{engine}' does not support multi-turn chat.")
        if caps.requires_documents_at_query:
            ui.info(f"Use 'fitz quickstart <folder> \"question\" --engine {engine}' for one-off queries.")
        else:
            ui.info(f"Use 'fitz query --engine {engine}' for single queries.")
        raise typer.Exit(0)

    # Engines with collection support use collection-based chat
    if caps.supports_collections:
        _run_collection_chat(collection, engine)
    else:
        ui.warning(f"Chat for engine '{engine}' requires collections which it doesn't support.")
        ui.info(f"Use 'fitz query --engine {engine}' for single queries.")
        raise typer.Exit(0)


def _run_collection_chat(collection: Optional[str], engine_name: str) -> None:
    """Run chat using an engine with collection support."""
    from fitz_ai.engines.classic_rag.pipeline.engine import RAGPipeline

    # Load config
    raw_config, typed_config = load_classic_rag_config()
    if typed_config is None:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)

    default_collection = typed_config.retrieval.collection

    # Collection selection
    if collection:
        selected_collection = collection
    else:
        collections = get_collections(raw_config)
        if not collections:
            ui.error("No collections found. Run 'fitz ingest' first to create a collection.")
            raise typer.Exit(1)
        elif len(collections) == 1:
            selected_collection = collections[0]
            ui.info(f"Using collection: {selected_collection}")
        else:
            print()
            selected_collection = ui.prompt_numbered_choice(
                "Collection", collections, default_collection
            )

    # Update config with selected collection
    typed_config.retrieval.collection = selected_collection

    # Create pipeline
    try:
        pipeline = RAGPipeline.from_config(typed_config)
    except Exception as e:
        ui.error(f"Failed to initialize pipeline: {e}")
        logger.exception("Pipeline initialization error")
        raise typer.Exit(1)

    # Chat loop
    _display_welcome(selected_collection, engine_name)

    history: List[Dict[str, str]] = []

    try:
        while True:
            # Get user input
            try:
                if RICH:
                    from rich.prompt import Prompt

                    user_input = Prompt.ask("\n[bold green]You[/bold green]")
                else:
                    user_input = input("\nYou: ").strip()
            except EOFError:
                break

            # Check for exit commands
            if user_input.lower() in ("exit", "quit", "q"):
                break

            # Skip empty input
            if not user_input.strip():
                continue

            # Retrieve chunks for current question
            try:
                chunks = pipeline.retrieval.retrieve(user_input)
            except Exception as e:
                ui.error(f"Retrieval failed: {e}")
                logger.exception("Retrieval error")
                continue

            # Build messages with history and context
            messages = _build_messages(history, chunks, user_input)

            # Get LLM response
            try:
                response = pipeline.chat.chat(messages)
            except Exception as e:
                ui.error(f"LLM request failed: {e}")
                logger.exception("LLM error")
                continue

            # Display response and sources
            _display_assistant_message(response)
            display_sources(chunks, indent=12)

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        pass  # Graceful exit on Ctrl+C

    _display_goodbye()
