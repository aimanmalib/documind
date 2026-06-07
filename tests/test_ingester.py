"""Tests for IngestAgent."""

from __future__ import annotations

import pytest

from documind.agents.ingester import IngestAgent, TextChunk, Document


class TestIngestAgent:
    """Test the IngestAgent."""

    @pytest.mark.asyncio
    async def test_ingest_markdown(self, config, mock_client, sample_doc_path):
        agent = IngestAgent(client=mock_client, config=config)
        doc = await agent.run(path=sample_doc_path)
        assert isinstance(doc, Document)
        assert doc.filename == "test_document.md"
        assert doc.file_type == "markdown"
        assert len(doc.chunks) > 0

    @pytest.mark.asyncio
    async def test_ingest_plain_text(self, config, mock_client, tmp_path):
        p = tmp_path / "plain.txt"
        p.write_text("Simple text content. " * 50)
        agent = IngestAgent(client=mock_client, config=config)
        doc = await agent.run(path=p)
        assert doc.file_type == "text"
        assert doc.total_words > 0

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_raises(self, config, mock_client):
        agent = IngestAgent(client=mock_client, config=config)
        with pytest.raises(FileNotFoundError):
            await agent.run(path="/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_ingest_creates_chunks(self, config, mock_client, sample_doc_path):
        agent = IngestAgent(client=mock_client, config=config)
        doc = await agent.run(path=sample_doc_path)
        for chunk in doc.chunks:
            assert chunk.chunk_id.startswith("doc-")
            assert chunk.document_id == doc.doc_id
            assert len(chunk.text) > 0

    @pytest.mark.asyncio
    async def test_ingest_with_sections(self, config, mock_client, sample_doc_path):
        agent = IngestAgent(client=mock_client, config=config)
        doc = await agent.run(path=sample_doc_path, extract_sections=True)
        sections = {c.section for c in doc.chunks}
        assert len(sections) > 1

    @pytest.mark.asyncio
    async def test_batch_ingest(self, config, mock_client, tmp_path):
        for i in range(3):
            p = tmp_path / f"doc{i}.txt"
            p.write_text(f"Document {i} content. " * 30)
        agent = IngestAgent(client=mock_client, config=config)
        docs = await agent.run_batch([tmp_path / f"doc{i}.txt" for i in range(3)])
        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_ingest_html(self, config, mock_client, tmp_path):
        p = tmp_path / "test.html"
        html_content = """
        <html>
        <head><title>Ignore it </title><style>body{color:red;}</style></head>
        <body>
              <h1>Main Title</h1>
              <p>Paragraph</p>
              <h2>Sub Title</h2>
              <div> some div content. <script> console.log("ignore)</script></div>
        </body>
        </html>
        """
        p.write_text(html_content)
        agent = IngestAgent(client=mock_client, config=config)
        doc = await agent.run(path=p, extract_sections=True)
        assert doc.file_type == "html"
        assert "# Main Title" in doc.content
        assert "## Sub Title" in doc.content
        assert "Paragraph" in doc.content
        assert "Ignore it" not in doc.content
        assert "body{color:red;}" not in doc.content
        assert "console.log" not in doc.content

        sections = {c.section for c in doc.chunks}
        assert "Main Title" in sections
        assert "Sub Title" in sections




class TestTextChunk:
    """Test TextChunk dataclass."""

    def test_word_count_computed(self):
        c = TextChunk(
            chunk_id="c1",
            document_id="d1",
            text="hello world test",
            start_offset=0,
            end_offset=16,
        )
        assert c.word_count == 3

    def test_char_count_computed(self):
        c = TextChunk(
            chunk_id="c1", document_id="d1", text="hello", start_offset=0, end_offset=5
        )
        assert c.char_count == 5


class TestDocument:
    """Test Document dataclass."""

    def test_total_words(self):
        chunks = [
            TextChunk("c1", "d1", "hello world", 0, 11),
            TextChunk("c2", "d1", "foo bar baz", 12, 23),
        ]
        doc = Document(
            doc_id="d1",
            filename="test.txt",
            content="...",
            file_type="text",
            chunks=chunks,
        )
        assert doc.total_words == 5
