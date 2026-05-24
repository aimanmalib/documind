"""DocuMind CLI — command-line interface for document Q&A."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from documind.config import DocuMindConfig
from documind.pipeline import DocuMindPipeline
from documind.token_tracker import TokenTracker

console = Console()


@click.group()
def main() -> None:
    """DocuMind — Document Q&A powered by Xiaomi MiMo V2.5 Pro."""


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def ingest(files: tuple[str, ...]) -> None:
    """Ingest one or more documents."""
    if not files:
        console.print("[red]No files provided.[/red]")
        sys.exit(1)

    async def _run() -> None:
        pipeline = DocuMindPipeline()
        try:
            results = await pipeline.ingest_documents(list(files))
            table = Table(title="Ingestion Results")
            table.add_column("File", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Chunks", justify="right")
            table.add_column("Words", justify="right")
            for r in results:
                table.add_row(r.filename, r.file_type, str(len(r.chunks)), str(r.total_words))
            console.print(table)
        finally:
            await pipeline.client.close()

    asyncio.run(_run())


@main.command()
@click.argument("query")
@click.option("--no-verify", is_flag=True, help="Skip fact-checking")
@click.option("--export", is_flag=True, help="Export session to file")
def ask(query: str, no_verify: bool, export: bool) -> None:
    """Ask a question about ingested documents."""
    console.print("[yellow]Ingest a document first, then ask questions.[/yellow]")


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True))
def summarize(files: tuple[str, ...]) -> None:
    """Summarize documents."""
    if not files:
        console.print("[red]No files provided.[/red]")
        sys.exit(1)

    async def _run() -> None:
        pipeline = DocuMindPipeline()
        try:
            for f in files:
                doc = await pipeline.ingest(f)
                summary = await pipeline.summarize_document(doc)
                console.print(Panel(Markdown(summary.summary_text), title=f"Summary: {doc.filename}"))
        finally:
            await pipeline.client.close()

    asyncio.run(_run())


@main.command()
def stats() -> None:
    """Show DocuMind token usage statistics."""
    table = Table(title="DocuMind — Daily Token Estimates")
    table.add_column("Agent", style="cyan")
    table.add_column("Est. Tokens/Day", justify="right", style="green")
    table.add_column("Share", justify="right")

    total = sum(TokenTracker.DAILY_ESTIMATES.values())
    for agent, tokens in TokenTracker.DAILY_ESTIMATES.items():
        share = tokens / total * 100 if total else 0
        table.add_row(agent, f"{tokens:.2f}M", f"{share:.1f}%")

    table.add_row("[bold]TOTAL[/]", f"[bold]{total:.2f}M[/]", "[bold]100%[/]")
    console.print(table)


if __name__ == "__main__":
    main()
