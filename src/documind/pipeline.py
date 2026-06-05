"""Pipeline orchestrator — coordinate agents for document Q&A."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from documind.agents import (
    IngestAgent,
    IndexAgent,
    RetrieverAgent,
    AnswerAgent,
    SummarizerAgent,
    FactCheckAgent,
    CitationAgent,
    ExportAgent,
)
from documind.agents.ingester import TextChunk, Document
from documind.agents.answerer import Answer
from documind.client import MiMoClient
from documind.token_tracker import TokenTracker


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    document: Document
    summary: Any
    answers: list[Answer] = field(default_factory=list)
    export_path: str = ""
    token_summary: dict[str, Any] = field(default_factory=dict)


class DocuMindPipeline:
    """Orchestrate the multi-agent document Q&A pipeline."""

    def __init__(self, config=None):
        from documind.config import DocuMindConfig

        self.config = config or DocuMindConfig.from_env()
        self.client = MiMoClient(self.config)
        self.tracker = TokenTracker()

        self.ingester = IngestAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )
        self.indexer = IndexAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )
        self.retriever = RetrieverAgent(
            client=self.client,
            config=self.config,
            tracker=self.tracker,
            index=self.indexer,
        )
        self.answerer = AnswerAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )
        self.summarizer = SummarizerAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )
        self.fact_checker = FactCheckAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )
        self.citation_agent = CitationAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )
        self.exporter = ExportAgent(
            client=self.client, config=self.config, tracker=self.tracker
        )

        self._documents: list[Document] = []
        self._chunks: list[TextChunk] = []
        self._session_answers: list[Answer] = []

    async def ingest_documents(self, paths: list[str]) -> list[Document]:
        """Ingest and index multiple documents."""
        docs = []
        for p in paths:
            doc = await self.ingester.run(path=p)
            docs.append(doc)
            self._chunks.extend(doc.chunks)
        self._documents.extend(docs)
        if self._chunks:
            await self.indexer.index_chunks(self._chunks)
        return docs

    async def ingest_text(self, text: str, name: str = "document") -> Document:
        """Ingest raw text as a document."""
        import tempfile

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"{name}_", delete=False
        )
        tmp.write(text)
        tmp.close()
        doc = await self.ingester.run(path=tmp.name)
        self._documents.append(doc)
        self._chunks.extend(doc.chunks)
        await self.indexer.index_chunks(doc.chunks)
        return doc

    async def ask(self, question: str, rerank: bool = False) -> Answer:
        """Ask a question against indexed documents."""
        chunks = await self.retriever.run(query=question, rerank=rerank)
        answer = await self.answerer.run(question=question, chunks=chunks)
        self._session_answers.append(answer)
        return answer

    async def ask_with_verification(self, question: str, rerank: bool = True):
        """Ask and fact-check the answer."""
        chunks = await self.retriever.run(query=question, rerank=rerank)
        answer = await self.answerer.run(question=question, chunks=chunks)
        check = await self.fact_checker.run(answer=answer, source_chunks=chunks)
        self._session_answers.append(answer)
        return answer, check

    async def summarize_document(self, doc: Document, max_length: str = "medium"):
        """Summarize a document."""
        return await self.summarizer.run(document=doc, max_length=max_length)

    async def export_session(self, path: str, fmt: str = "markdown") -> str:
        """Export all session answers."""
        if not self._session_answers:
            return ""
        result = await self.exporter.run(
            answers=self._session_answers, output_path=path, format=fmt
        )
        return str(result)

    async def run_full(self, doc_path: str, questions: list[str] | None = None):
        """Run the complete pipeline."""
        doc = await self.ingester.run(path=doc_path)
        self._documents.append(doc)
        self._chunks.extend(doc.chunks)
        await self.indexer.index_chunks(doc.chunks)

        summary = await self.summarizer.run(document=doc)
        answers = []
        for q in questions or []:
            ans = await self.ask(q)
            answers.append(ans)

        export_path = ""
        if answers:
            export_path = await self.export_session(f"/tmp/documind-{doc.doc_id}.md")

        await self.client.close()
        return PipelineResult(
            document=doc,
            summary=summary,
            answers=answers,
            export_path=export_path,
            token_summary=self.tracker.summary(),
        )
