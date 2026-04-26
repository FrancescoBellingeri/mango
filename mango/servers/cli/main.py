"""Mango CLI entry point.

Usage:
    mango                                        # reads config from env
    mango --provider anthropic --model claude-sonnet-4-6
    mango --provider gemini --model gemini-2.5-pro-preview-05-06
    mango --provider openai --model gpt-4o
    mango --uri mongodb://... --no-schema
    mango train --file knowledge.jsonl           # bulk-load training data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from mango.agent.agent import MangoAgent
from mango.integrations.mongodb import MongoRunner
from mango.llm.factory import PROVIDERS, build_llm
from mango.integrations.chromadb import ChromaAgentMemory
from mango.memory.models import TrainingEntry
from mango.memory import make_entry_id
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


async def _load_training_file(memory: ChromaAgentMemory, file_path: str) -> None:
    """Bulk-load training data from a JSONL file into the training collection."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red]File not found:[/red] {file_path}")
        return

    imported = 0
    errors = 0
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                console.print(f"[yellow]Line {lineno}: invalid JSON — {exc}[/yellow]")
                errors += 1
                continue

            entry_type = raw.get("type", "training")
            try:
                if entry_type in ("training", "tool"):
                    entry = TrainingEntry(
                        id=raw.get("id", make_entry_id()),
                        question=raw["question"],
                        tool_name=raw["tool_name"],
                        tool_args=raw.get("tool_args", {}),
                        result_summary=raw.get("result_summary", ""),
                    )
                    await memory.train(entry)
                    imported += 1
                elif entry_type == "text":
                    await memory.save_text(raw["text"])
                    imported += 1
                else:
                    console.print(f"[yellow]Line {lineno}: unknown type '{entry_type}' — skipped[/yellow]")
                    errors += 1
            except KeyError as exc:
                console.print(f"[yellow]Line {lineno}: missing field {exc} — skipped[/yellow]")
                errors += 1

    console.print(
        f"[green]Training complete:[/green] {imported} entries loaded"
        + (f", {errors} skipped" if errors else "")
        + f"  (training total: {memory.training_count()})"
    )


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
    console.print(
        f"[dim]Memory: {memory.count()} interactions, "
        f"{memory.training_count()} training entries ({memory_dir})[/dim]"
    )

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

    # Single persistent event loop for the entire REPL session.
    # Using one loop avoids ChromaDB Rust/tokio backend confusion when
    # asyncio.run() would otherwise create/destroy separate loops per call.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _run(coro):
        return loop.run_until_complete(coro)

    # REPL loop.
    try:
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

            if question in {"/memory", "/remember"}:
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_row("Interactions:", str(memory.count()))
                table.add_row("Training entries:", str(memory.training_count()))
                table.add_row("Directory:", memory_dir)
                console.print(table)
                continue

            if question.startswith("/train"):
                parts = question.split(maxsplit=1)
                if len(parts) < 2:
                    console.print(
                        "[dim]Usage: /train <file.jsonl>[/dim]\n"
                        "[dim]JSONL format per line:[/dim]\n"
                        '[dim]  {"type": "training", "question": "...", "tool_name": "...", "tool_args": {...}}[/dim]\n'
                        '[dim]  {"type": "text", "text": "..."}[/dim]'
                    )
                    continue
                _run(_load_training_file(memory, parts[1].strip()))
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
                response = _run(agent.ask(question, on_tool_call=on_tool_call))
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
    finally:
        loop.close()


def _add_common_args(p: "argparse.ArgumentParser") -> None:
    p.add_argument("--uri", default=os.getenv("MONGODB_URI"), help="MongoDB URI (default: $MONGODB_URI)")
    p.add_argument("--memory-dir", default=os.getenv("MANGO_MEMORY_DIR", ".mango_memory"), help="ChromaDB directory")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mango",
        description="Mango — natural language interface for MongoDB",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── mango chat (default) ──────────────────────────────────────────────
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat (default)")
    _add_common_args(chat_parser)
    chat_parser.add_argument("--provider", default=os.getenv("MANGO_PROVIDER", "openai"), choices=PROVIDERS)
    chat_parser.add_argument("--model", default=os.getenv("MANGO_MODEL", "gpt-5.4"))
    chat_parser.add_argument("--api-key", default=None)
    chat_parser.add_argument("--no-schema", action="store_true", help="Skip schema introspection")
    chat_parser.add_argument("--no-memory", action="store_true", help="Disable memory layer")
    chat_parser.add_argument("--verbose", "-v", action="store_true")

    # ── mango train ───────────────────────────────────────────────────────
    train_parser = subparsers.add_parser("train", help="Bulk-load training data into memory")
    _add_common_args(train_parser)
    train_parser.add_argument("--file", "-f", required=True, help="JSONL file to import")

    # ── mango export ──────────────────────────────────────────────────────
    export_parser = subparsers.add_parser("export", help="Export all memory to a JSONL file")
    _add_common_args(export_parser)
    export_parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")

    # Legacy: no subcommand → behave as 'chat' for backwards compat.
    parser.add_argument("--uri", default=os.getenv("MONGODB_URI"))
    parser.add_argument("--provider", default=os.getenv("MANGO_PROVIDER", "openai"), choices=PROVIDERS)
    parser.add_argument("--model", default=os.getenv("MANGO_MODEL"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--memory-dir", default=os.getenv("MANGO_MEMORY_DIR", ".mango_memory"))
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # ── train subcommand ──────────────────────────────────────────────────
    if args.command == "train":
        memory = ChromaAgentMemory(persist_dir=args.memory_dir)
        asyncio.run(_load_training_file(memory, args.file))
        return

    # ── export subcommand ─────────────────────────────────────────────────
    if args.command == "export":
        memory = ChromaAgentMemory(persist_dir=args.memory_dir)

        async def _export() -> None:
            entries = await memory.export_all()
            output = Path(args.output)
            with output.open("w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            console.print(f"[green]Exported {len(entries)} entries to {args.output}[/green]")

        asyncio.run(_export())
        return

    # ── chat (default) ────────────────────────────────────────────────────
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
