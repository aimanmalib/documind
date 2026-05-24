"""Shared fixtures for DocuMind tests."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from documind.client import CompletionResponse, MiMoClient
from documind.config import DocuMindConfig
from documind.token_tracker import TokenTracker


@pytest.fixture
def config():
    """Test configuration with mock API key."""
    return DocuMindConfig(
        mimo_api_key="test-key-123",
        mimo_base_url="https://test.api.example.com/v1",
        mimo_model="mimo-v2.5-pro",
        max_chunk_size=256,
        chunk_overlap=32,
        max_retrieval_k=3,
        citation_format="apa",
        output_dir="/tmp/documind-test-output",
    )


@pytest.fixture
def tracker():
    """Fresh token tracker."""
    return TokenTracker()


@pytest.fixture
def mock_response():
    """Factory for mock completion responses."""
    def _make(
        content: str = "Test response",
        reasoning: str = "",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> CompletionResponse:
        return CompletionResponse(
            content=content,
            reasoning_content=reasoning,
            model="mimo-v2.5-pro",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            latency_ms=250.0,
        )
    return _make


@pytest.fixture
def mock_client(mock_response):
    """Mock MiMo client that returns canned responses."""
    client = AsyncMock(spec=MiMoClient)
    client.chat_completion = AsyncMock(return_value=mock_response())
    client.stats = {"request_count": 0, "total_tokens": 0}
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_text():
    """Sample document text for testing."""
    return """# Test Document

This is a test document for the DocuMind pipeline. It contains multiple paragraphs
and sections to test chunking, indexing, and retrieval.

## Section 1: Introduction

Document analysis is the process of extracting useful information from text documents.
Natural language processing techniques are used to understand the content and structure.

## Section 2: Methods

The pipeline uses multiple agents working together. First, the ingester parses the document.
Then the indexer builds a searchable index. Finally, the retriever finds relevant chunks.

## Section 3: Results

The system achieves high accuracy in question answering tasks. Token usage is tracked
per agent for cost optimization. The MiMo model provides excellent reasoning capabilities.

## Section 4: Conclusion

DocuMind demonstrates effective multi-agent document Q&A. Future work includes
adding more document formats and improving retrieval accuracy.
"""


@pytest.fixture
def sample_doc_path(sample_text, tmp_path):
    """Write sample text to a temporary file."""
    p = tmp_path / "test_document.md"
    p.write_text(sample_text)
    return p


@pytest.fixture
def sample_chunks():
    """Pre-built text chunks for testing."""
    from documind.agents.ingester import TextChunk
    return [
        TextChunk(
            chunk_id="doc-abc-chunk-0000",
            document_id="doc-abc",
            text="Document analysis is the process of extracting useful information from text.",
            start_offset=0,
            end_offset=80,
            section="Introduction",
        ),
        TextChunk(
            chunk_id="doc-abc-chunk-0001",
            document_id="doc-abc",
            text="The pipeline uses multiple agents working together for document processing.",
            start_offset=81,
            end_offset=155,
            section="Methods",
        ),
        TextChunk(
            chunk_id="doc-abc-chunk-0002",
            document_id="doc-abc",
            text="The system achieves high accuracy with the MiMo model for reasoning tasks.",
            start_offset=156,
            end_offset=232,
            section="Results",
        ),
    ]
