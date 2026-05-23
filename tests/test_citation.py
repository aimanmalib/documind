"""Tests for CitationAgent."""
from __future__ import annotations

import pytest

from documind.agents.citation import CitationAgent, FormattedCitation


class TestCitationAgent:
    """Test the CitationAgent."""

    def test_format_apa(self, config, mock_client, tracker):
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        result = agent.format_citation("Test Title", "Author Name", "2024", source_num=1)
        assert isinstance(result, FormattedCitation)
        assert result.style == "apa"
        assert "Author Name" in result.inline

    def test_format_mla(self, config, mock_client, tracker):
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        result = agent.format_citation("Title", "Author", "2024", style="mla", source_num=1)
        assert result.style == "mla"

    def test_format_chicago(self, config, mock_client, tracker):
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        result = agent.format_citation("Title", "Author", "2024", style="chicago", source_num=1)
        assert result.style == "chicago"

    @pytest.mark.asyncio
    async def test_generate_references(self, config, mock_client, tracker):
        from documind.client import CompletionResponse
        mock_client.chat_completion.return_value = CompletionResponse(
            content="1. Author (2024). Title.", prompt_tokens=50, completion_tokens=20, total_tokens=70,
        )
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        refs = await agent.generate_references([{"source_num": "1", "chunk_id": "c1", "document_id": "d1", "excerpt": "test"}])
        assert isinstance(refs, str)

    @pytest.mark.asyncio
    async def test_run_with_citations(self, config, mock_client, tracker):
        from documind.client import CompletionResponse
        mock_client.chat_completion.return_value = CompletionResponse(
            content="1. Ref.", prompt_tokens=50, completion_tokens=10, total_tokens=60,
        )
        agent = CitationAgent(client=mock_client, config=config, tracker=tracker)
        result = await agent.run(citations=[{"source_num": "1"}])
        assert isinstance(result, str)
