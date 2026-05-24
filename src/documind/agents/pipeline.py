"""Pipeline orchestrator — coordinates all agents."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .answerer import Answer, AnswerAgent
from .citation import CitationAgent
from .client import MiMoClient
from .config import DocuMindConfig
from .export import ExportAgent
from .fact_checker import FactCheckAgent, FactCheckResult
from .indexer import IndexAgent
from .ingester import Document, IngestAgent
from .retriever import RetrieverAgent, RetrievalResult
from .summarizer import SummarizerAgent, Summary
from .token_tracker import TokenTracker


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""

    document: Document
    summary: Summary
    answers: list[Answer] = field(default_factory=list)
    fact_checks: list[FactCheckResult] = field(default_factory=list)
    export_path: str = ""
    token_summary: dict[str, Any] = field(default_factory=dict)


class DocuMindPipeline:
    """Orchestrate the full DocuMind document Q&A pipeline."""

    def __init__(self, config: DocuMindConfig | None = None) -> None:
        self.config = config or DocuMindConfig.from_env()
        self.client = MiMoClient(self.config)
        self.tracker = TokenTracker()

        self.ingester = IngestAgent(client=self.client, config=self.config, tracker=self.tracker)
        self.indexer = IndexAgent(client=self.client, config=self.config, tracker=self.tracker)
        self.retriever = RetrieverAgent(
            client=self.client, config=self.config, tracker=self.tracker, index=self.indexer
        )
        self.answerer = AnswerAgent(client=self.client, config=self.config, tracker=self.tracker)
        self.summarizer = SummarizerAgent(client=self.client, config=self.config, tracker=self.tracker)
        self.fact_checker = FactCheckAgent(client=self.client, config=self.config, tracker=self.tracker)
        self.citation_agent = CitationAgent(client=self.client, config=self.config, tracker=self.tracker)
        self.exporter = ExportAgent(client=self.client, config=self.config, tracker=self.tracker)

    async def ingest(self, path: str | Path) -> Document:
        """Ingest and index a document."""
        doc = await self.ingester.run(path=path)
        await self.indexer.run(chunks=doc.chunks)
        return doc

    async def ask(self, question: str, *, rerank: bool = False) -> Answer:
        """Ask a question against the indexed documents."""
        chunks = await self.retriever.run(query=question, rerank=rerank)
        answer = await self.answerer.run(question=question, chunks=chunks)
        return answer

    async def ask_with_verification(
        self, question: str, *, rerank: bool = True
    ) -> tuple[Answer, FactCheckResult]:
        """Ask a question and verify the answer."""
        chunks = await self.retriever.run(query=question, rerank=rerank)
        answer = await self.answerer.run(question=question, chunks=chunks)
        fact_check = await self.fact_checker.run(answer=answer, source_chunks=chunks)
        return answer, fact_check

    async def summarize_doc(self, document: Document, max_length: str = "medium") -> Summary:
        """Summarize an ingested document."""
        return await self.summarizer.run(document=document, max_length=max_length)

    async def run_full_pipeline(
        self,
        doc_path: str | Path,
        questions: list[str] | None = None,
        *,
        export: bool = True,
        export_format: str = "markdown",
    ) -> PipelineResult:
        """Run the complete pipeline: ingest → index → summarize → answer → export."""
        doc = await self.ingest(doc_path)
        summary = await self.summarize_doc(doc)

        answers: list[Answer] = []
        fact_checks: list[FactCheckResult] = []

        if questions:
            for q in questions:
                ans, check = await self.ask_with_verification(q)
                answers.append(ans)
                fact_checks.append(check)

        export_path = ""
        if export and answers:
            out_dir = Path(self.config.output_dir)
            out_file = out_dir / f"{doc.doc_id}_qa_session"
            p = await self.exporter.run(
                answers=answers,
                output_path=str(out_file),
                format=export_format,
                title=f"DocuMind Q&A: {doc.filename}",
            )
            export_path = str(p)

        await self.client.close()

        return PipelineResult(
            document=doc,
            summary=summary,
            answers=answers,
            fact_checks=fact_checks,
            export_path=export_path,
            token_summary=self.tracker.summary(),
        )
