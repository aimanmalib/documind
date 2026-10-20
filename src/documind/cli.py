"""DocuMind CLI — command-line interface for document Q&A."""

from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from documind import __version__
from documind.pipeline import DocuMindPipeline
from documind.token_tracker import TokenTracker

console = Console()


def _print_usage(tracker: TokenTracker) -> None:
    """Print the token usage actually reported by the LLM for this run."""
    by_agent = tracker.by_agent()
    if not by_agent:
        return
    table = Table(title="Token usage (this run)")
    table.add_column("Agent", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Prompt", justify="right")
    table.add_column("Completion", justify="right")
    table.add_column("Total", justify="right", style="green")
    for agent, u in by_agent.items():
        table.add_row(
            agent,
            str(u["calls"]),
            str(u["prompt"]),
            str(u["completion"]),
            str(u["total"]),
        )
    table.add_row(
        "[bold]TOTAL[/]",
        str(tracker.total_calls()),
        "",
        "",
        f"[bold]{tracker.total_tokens()}[/]",
    )
    console.print(table)


@click.group()
@click.version_option(version=__version__, prog_name="documind")
def main() -> None:
    """DocuMind — multi-agent document Q&A for any OpenAI-compatible LLM.

    Configure the LLM with DOCUMIND_PROVIDER (default: openai) and the
    provider's API key env var, e.g. OPENAI_API_KEY. For any other
    OpenAI-compatible endpoint set DOCUMIND_BASE_URL, DOCUMIND_API_KEY and
    DOCUMIND_MODEL.
    """


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def ingest(files: tuple[str, ...]) -> None:
    """Parse, chunk and index documents (no LLM calls).

    Useful to check how a file is chunked before asking questions.
    """
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
                table.add_row(
                    r.filename, r.file_type, str(len(r.chunks)), str(r.total_words)
                )
            console.print(table)
        finally:
            await pipeline.client.close()

    asyncio.run(_run())


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-q",
    "--question",
    "questions",
    multiple=True,
    help="Question to ask (repeat for several questions).",
)
@click.option("--verify", is_flag=True, help="Fact-check each answer against sources.")
@click.option("--rerank", is_flag=True, help="Let the LLM rerank retrieved chunks.")
@click.option(
    "--export",
    "export_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the Q&A session to this file.",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    show_default=True,
    help="Export format.",
)
def ask(
    files: tuple[str, ...],
    questions: tuple[str, ...],
    verify: bool,
    rerank: bool,
    export_path: str | None,
    export_format: str,
) -> None:
    """Ingest FILES and answer questions about them, with citations."""
    if not files:
        console.print("[red]No files provided.[/red]")
        sys.exit(1)
    if not questions:
        console.print("[red]Provide at least one question with -q.[/red]")
        sys.exit(1)

    async def _run() -> None:
        pipeline = DocuMindPipeline()
        try:
            docs = await pipeline.ingest_documents(list(files))
            for d in docs:
                console.print(
                    f"[green]✓[/] Ingested {d.filename} ({len(d.chunks)} chunks)"
                )
            for q in questions:
                console.print(f"\n[bold yellow]Q:[/] {q}")
                if verify:
                    ans, check = await pipeline.ask_with_verification(q, rerank=rerank)
                else:
                    ans = await pipeline.ask(q, rerank=rerank)
                console.print(Markdown(ans.answer_text))
                if verify:
                    color = {
                        "verified": "green",
                        "partially_verified": "yellow",
                    }.get(check.overall_verdict, "red")
                    console.print(
                        f"[{color}]Verification: {check.overall_verdict}[/] "
                        f"({check.claims_verified}/{check.claims_checked} claims verified)"
                    )
            if export_path:
                path = await pipeline.export_session(export_path, fmt=export_format)
                console.print(f"\n[blue]Exported to:[/] {path}")
            console.print()
            _print_usage(pipeline.tracker)
        finally:
            await pipeline.client.close()

    asyncio.run(_run())


@main.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def summarize(files: tuple[str, ...]) -> None:
    """Summarize documents."""
    if not files:
        console.print("[red]No files provided.[/red]")
        sys.exit(1)

    async def _run() -> None:
        pipeline = DocuMindPipeline()
        try:
            for doc in await pipeline.ingest_documents(list(files)):
                summary = await pipeline.summarize_document(doc)
                console.print(
                    Panel(
                        Markdown(summary.summary_text), title=f"Summary: {doc.filename}"
                    )
                )
            _print_usage(pipeline.tracker)
        finally:
            await pipeline.client.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
