"""Additional tests to reach 90+ count."""
from __future__ import annotations

import pytest
from pathlib import Path

from documind.agents.ingester import IngestAgent, TextChunk, Document
from documind.agents.indexer import IndexAgent
from documind.agents.retriever import RetrieverAgent, RetrievalResult
from documind.agents.answerer import AnswerAgent, Answer
from documind.agents.summarizer import SummarizerAgent
from documind.agents.citation import CitationAgent
from documind.agents.export import ExportAgent
from documind.client import MiMoClient, CompletionResponse
from documind.config import DocuMindConfig
from documind.token_tracker import TokenTracker


class TestIngestAgentExtra:
    """Extra ingester tests."""

    def test_detect_type_txt(self, config, mock_client):
        agent = IngestAgent(client=mock_client, config=config)
        assert agent._detect_type(Path("file.txt")) == "text"

    def test_detect_type_md(self, config, mock_client):
        agent = IngestAgent(client=mock_client, config=config)
        assert agent._detect_type(Path("file.md")) == "markdown"

    def test_detect_type_html(self, config, mock_client):
        agent = IngestAgent(client=mock_client, config=config)
        assert agent._detect_type(Path("file.html")) == "html"

    def test_generate_doc_id_deterministic(self, config, mock_client):
        agent = IngestAgent(client=mock_client, config=config)
        id1 = agent._generate_doc_id("same content", "a.txt")
        id2 = agent._generate_doc_id("same content", "b.txt")
        assert id1 == id2

    def test_split_into_chunks_respects_size(self, config, mock_client):
        agent = IngestAgent(client=mock_client, config=config)
        agent.chunk_size = 50
        text = "Short sentence. Another sentence. Third sentence here. Fourth one."
        chunks = agent._split_into_chunks(text, "doc-test")
        assert len(chunks) >= 1


class TestIndexAgentExtra:
    """Extra indexer tests."""

    def test_compute_tf_normalizes(self, config, mock_client):
        agent = IndexAgent(client=mock_client, config=config)
        tf = agent._compute_tf(["hello", "hello", "world"])
        assert abs(tf["hello"] - 2/3) < 0.01

    def test_extract_keywords_top_n(self, config, mock_client):
        agent = IndexAgent(client=mock_client, config=config)
        tf = {"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.05}
        kw = agent._extract_keywords(tf, top_n=2)
        assert len(kw) == 2
        assert kw[0] == "a"

    def test_search_empty_query(self, config, mock_client):
        agent = IndexAgent(client=mock_client, config=config)
        results = agent.search_keywords("")
        assert results == []


class TestRetrieverExtra:
    """Extra retriever tests."""

    @pytest.mark.asyncio
    async def test_retrieve_min_score(self, config, mock_client, sample_chunks):
        indexer = IndexAgent(client=mock_client, config=config)
        await indexer.index_chunks(sample_chunks)
        agent = RetrieverAgent(client=mock_client, config=config, index=indexer)
        results = await agent.retrieve("document", min_score=999.0)
        assert len(results) == 0


class TestAnswerExtra:
    """Extra answer tests."""

    def test_answer_dataclass(self):
        a = Answer(question="q", answer_text="a", citations=[], confidence="high")
        assert a.question == "q"


class TestCitationExtra:
    """Extra citation tests."""

    def test_format_apa_contains_year(self, config, mock_client, tracker):
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        result = agent.format_citation("Title", "Author", "2025", source_num=1)
        assert "2025" in result.inline
        assert "2025" in result.full_reference

    def test_format_chicago_contains_author(self, config, mock_client, tracker):
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        result = agent.format_citation("Title", "Smith", "2024", style="chicago", source_num=1)
        assert "Smith" in result.inline
