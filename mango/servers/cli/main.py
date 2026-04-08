"""Mango CLI entry point.

Usage:
    mango                                        # reads config from env
    mango --provider anthropic --model claude-sonnet-4-6
    mango --provider gemini --model gemini-2.5-pro-preview-05-06
    mango --provider openai --model gpt-4o
    mango --uri mongodb://... --no-schema
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from mango.agent.agent import MangoAgent
from mango.integrations.mongodb import MongoRunner
from mango.llm.factory import PROVIDERS, build_llm
from mango.integrations.chromadb import ChromaAgentMemory
from mango.tools.base import ToolRegistry
from mango.tools.mongo_tools import build_mongo_tools

load_dotenv()

console = Console()


def _setup_logging(verbose: bool) -> None:
    for noisy in ("pymongo", "chromadb", "httpx", "httpcore", "openai", "anthropic", "markdown_it", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
        logging.getLogger("mango").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)


def _print_banner(db_name: str, provider: str, model: str, n_collections: int) -> None:
    text = (
        f"[bold green]Mango[/bold green] — MongoDB AI assistant\n"
        f"Database: [cyan]{db_name}[/cyan]  |  "
        f"Collections: [cyan]{n_collections}[/cyan]  |  "
        f"Provider: [cyan]{provider}[/cyan]  |  "
        f"Model: [cyan]{model}[/cyan]\n"
        f"Type [bold]exit[/bold] or [bold]quit[/bold] to leave. "
        f"Type [bold]/reset[/bold] to clear conversation history."
    )
    console.print(Panel(text, border_style="green"))


def run(
    uri: str,
    provider: str,
    model: str | None,
    api_key: str | None,
    introspect: bool,
    verbose: bool,
    memory_dir: str = ".mango_memory",
) -> None:
    _setup_logging(verbose)

    # Connect backend.
    backend = MongoRunner()
    with console.status("[bold green]Connecting to MongoDB…"):
        try:
            backend.connect(uri)
        except Exception as exc:
            console.print(f"[red]Connection failed:[/red] {exc}")
            sys.exit(1)

    collections = backend.list_collections()
    db_name = backend._database.name

    # Introspect schema at startup (optional).
    schema = None
    if introspect:
        with console.status(f"[bold green]Introspecting {len(collections)} collections…"):
            try:
                schema = backend.introspect_schema()
            except Exception as exc:
                console.print(f"[yellow]Schema introspection failed:[/yellow] {exc}")

    # Build LLM service via factory.
    try:
        llm = build_llm(provider=provider, model=model, api_key=api_key)
    except Exception as exc:
        console.print(f"[red]LLM init failed:[/red] {exc}")
        sys.exit(1)

    # Initialise memory.
    memory = ChromaAgentMemory(persist_dir=memory_dir)
    console.print(f"[dim]Memory: {memory.count()} stored interactions ({memory_dir})[/dim]")

    # Register tools.
    registry = ToolRegistry()
    for tool in build_mongo_tools(backend):
        registry.register(tool)

    # Build agent.
    agent = MangoAgent(
        llm_service=llm,
        tool_registry=registry,
        db=backend,
        agent_memory=memory,
        schema=schema,
        introspect=introspect,
    )
    agent.setup()

    _print_banner(db_name, provider, llm.get_model_name(), len(collections))

    # REPL loop.
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        if question == "/reset":
            agent.reset_conversation()
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if question == "/remember":
            console.print(f"[dim]Memory entries: {memory.count()} — memory is managed automatically by the agent.[/dim]")
            continue

        if question == "/memory":
            console.print(f"[dim]Memory entries: {memory.count()}[/dim]")
            continue

        def on_tool_call(tool_name: str, tool_args: dict, result_text: str) -> None:
            args_preview = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
            result_preview = result_text[:120].replace("\n", " ")
            if len(result_text) > 120:
                result_preview += "…"
            console.print(
                f"  [dim]⚙[/dim] [bold]{tool_name}[/bold]"
                f"[dim]({args_preview})[/dim]"
                f"  [dim]→ {result_preview}[/dim]"
            )

        try:
            with console.status("[bold green]Thinking…"):
                pass
            response = asyncio.run(agent.ask(question, on_tool_call=on_tool_call))
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            if verbose:
                console.print_exception()
            continue

        console.print("\n[bold green]Mango[/bold green]")
        console.print(Markdown(response.answer))

        if verbose:
            meta = Text(
                f"  iterations={response.iterations}  "
                f"in={response.input_tokens}tok  out={response.output_tokens}tok"
                + (f"  memory_hits={response.memory_hits}" if response.memory_hits else ""),
                style="dim",
            )
            console.print(meta)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mango",
        description="Mango — natural language interface for MongoDB",
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("MONGODB_URI"),
        help="MongoDB connection URI (default: $MONGODB_URI)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("MANGO_PROVIDER", "openai"),
        choices=PROVIDERS,
        help="LLM provider: openai | anthropic | gemini (default: $MANGO_PROVIDER or openai)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MANGO_MODEL"),
        help="Model ID override. Uses provider default if omitted.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Falls back to provider env var if omitted.",
    )
    parser.add_argument(
        "--no-schema",
        action="store_true",
        help="Skip schema introspection at startup",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable memory layer (no ChromaDB)",
    )
    parser.add_argument(
        "--memory-dir",
        default=os.getenv("MANGO_MEMORY_DIR", ".mango_memory"),
        help="Directory for ChromaDB persistence (default: .mango_memory)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging and show token usage",
    )

    args = parser.parse_args()

    if not args.uri:
        console.print("[red]Error:[/red] MongoDB URI required. Set --uri or $MONGODB_URI.")
        sys.exit(1)

    run(
        uri=args.uri,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        introspect=not args.no_schema,
        memory_dir=":memory:" if args.no_memory else args.memory_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
