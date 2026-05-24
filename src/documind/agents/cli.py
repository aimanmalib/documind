"""Click CLI for DocuMind — Document Q&A pipeline."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .config import DocuMindConfig
from .pipeline import DocuMindPipeline

console = Console()


def _run(coro):
    """Run an async function."""
    return asyncio.run(coro)


@click.group()
@click.version_option(version="1.0.0", prog_name="documind")
def main():
    """DocuMind — Document Q&A powered by Xiaomi MiMo V2.5 Pro."""
    pass


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--summarize/--no-summarize", default=True, help="Generate summary")
@click.option("--max-length", type=click.Choice(["short", "medium", "long"]), default="medium")
def ingest(file: str, summarize: bool, max_length: str):
    """Ingest a document and optionally summarize it."""
    config = DocuMindConfig.from_env()
    pipeline = DocuMindPipeline(config)

    console.print(f"[bold blue]Ingesting:[/] {file}")

    async def _ingest():
        doc = await pipeline.ingest(file)
        console.print(f"[green]✓[/] Document loaded: {doc.filename}")
        console.print(f"  Type: {doc.file_type} | Chunks: {len(doc.chunks)} | Words: {doc.total_words}")

        if summarize:
            summary = await pipeline.summarize_doc(doc, max_length=max_length)
            console.print()
            console.print(Panel(Markdown(summary.summary_text), title="Summary", border_style="green"))
            if summary.key_points:
                console.print("[bold]Key Points:[/]")
                for kp in summary.key_points:
                    console.print(f"  • {kp}")

        await pipeline.client.close()
        return doc

    _run(_ingest())


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-q", "--question", multiple=True, help="Questions to ask")
@click.option("--rerank/--no-rerank", default=False, help="Use MiMo reranking")
@click.option("--verify/--no-verify", default=False, help="Fact-check answers")
@click.option("--export-format", type=click.Choice(["markdown", "json"]), default="markdown")
def ask(file: str, question: tuple[str, ...], rerank: bool, verify: bool, export_format: str):
    """Ask questions about a document."""
    if not question:
        console.print("[red]Error:[/] Provide at least one question with -q")
        sys.exit(1)

    config = DocuMindConfig.from_env()
    pipeline = DocuMindPipeline(config)

    async def _ask():
        doc = await pipeline.ingest(file)
        console.print(f"[green]✓[/] Ingested: {doc.filename} ({len(doc.chunks)} chunks)")

        answers = []
        for q in question:
            console.print(f"\n[bold yellow]Q:[/] {q}")
            if verify:
                ans, check = await pipeline.ask_with_verification(q, rerank=rerank)
                console.print(Markdown(ans.answer_text))
                verdict_color = {"verified": "green", "partially_verified": "yellow"}.get(check.overall_verdict, "red")
                console.print(f"[{verdict_color}]Verification: {check.overall_verdict}[/] ({check.claims_verified}/{check.claims_checked} verified)")
            else:
                ans = await pipeline.ask(q, rerank=rerank)
                console.print(Markdown(ans.answer_text))
            answers.append(ans)

        if export_format:
            from .export import ExportAgent
            exporter = ExportAgent(client=pipeline.client, config=config, tracker=pipeline.tracker)
            path = await exporter.run(
                answers=answers,
                output_path=f"{config.output_dir}/{doc.doc_id}_qa",
                format=export_format,
            )
            console.print(f"\n[blue]📄 Exported to:[/] {path}")

        await pipeline.client.close()

    _run(_ask())


@main.command()
def stats():
    """Show DocuMind token usage statistics."""
    table = Table(title="DocuMind — Daily Token Estimates")
    table.add_column("Agent", style="cyan")
    table.add_column("Est. Tokens/Day", justify="right", style="green")
    table.add_column("Share", justify="right")

    total = sum(DocuMindPipeline.tracker.DAILY_ESTIMATES.values() if hasattr(DocuMindPipeline, 'tracker') else TokenTracker.DAILY_ESTIMATES.values())
    from .token_tracker import TokenTracker
    for agent, tokens in TokenTracker.DAILY_ESTIMATES.items():
        share = tokens / total * 100 if total else 0
        table.add_row(agent, f"{tokens:.2f}M", f"{share:.1f}%")

    table.add_row("[bold]TOTAL[/]", f"[bold]{total:.2f}M[/]", "[bold]100%[/]")
    console.print(table)


if __name__ == "__main__":
    main()
