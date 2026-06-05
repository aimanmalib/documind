"""Tests for DocuMindPipeline."""

from __future__ import annotations


from documind.pipeline import DocuMindPipeline, PipelineResult


class TestDocuMindPipeline:
    """Test the pipeline orchestrator."""

    def test_init_creates_all_agents(self, config):
        pipeline = DocuMindPipeline(config)
        assert pipeline.ingester is not None
        assert pipeline.indexer is not None
        assert pipeline.retriever is not None
        assert pipeline.answerer is not None
        assert pipeline.summarizer is not None
        assert pipeline.fact_checker is not None
        assert pipeline.citation_agent is not None
        assert pipeline.exporter is not None

    def test_init_uses_config(self, config):
        pipeline = DocuMindPipeline(config)
        assert pipeline.config.mimo_api_key == "test-key-123"

    def test_init_default_config(self, monkeypatch):
        monkeypatch.setenv("MIMO_API_KEY", "env-key")
        pipeline = DocuMindPipeline()
        assert pipeline.config.mimo_api_key == "env-key"

    def test_tracker_shared_across_agents(self, config):
        pipeline = DocuMindPipeline(config)
        assert pipeline.tracker is not None


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_creation(self, sample_doc_path, config, mock_client):
        from documind.agents.ingester import Document, TextChunk
        from documind.agents.summarizer import Summary

        doc = Document(
            "d1",
            "t.txt",
            "content",
            "text",
            chunks=[TextChunk("c1", "d1", "content", 0, 7)],
        )
        summary = Summary("d1", "t.txt", "summary", [], 1, 0.5)
        result = PipelineResult(document=doc, summary=summary)
        assert result.document.doc_id == "d1"
        assert len(result.answers) == 0
