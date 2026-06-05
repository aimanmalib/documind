"""Tests for DocuMindConfig."""

from __future__ import annotations


from documind.config import DocuMindConfig


class TestDocuMindConfig:
    """Test configuration management."""

    def test_default_values(self):
        cfg = DocuMindConfig(mimo_api_key="test")
        assert cfg.mimo_model == "mimo-v2.5-pro"
        assert cfg.max_chunk_size == 512

    def test_validate_no_key(self):
        cfg = DocuMindConfig(mimo_api_key="")
        errors = cfg.validate()
        assert any("MIMO_API_KEY" in e for e in errors)

    def test_validate_bad_chunk_overlap(self):
        cfg = DocuMindConfig(mimo_api_key="k", max_chunk_size=100, chunk_overlap=200)
        errors = cfg.validate()
        assert any("chunk_overlap" in e for e in errors)

    def test_validate_good_config(self):
        cfg = DocuMindConfig(mimo_api_key="k")
        errors = cfg.validate()
        assert len(errors) == 0

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MIMO_API_KEY", "env-key")
        cfg = DocuMindConfig.from_env()
        assert cfg.mimo_api_key == "env-key"

    def test_citation_format_validation(self):
        cfg = DocuMindConfig(mimo_api_key="k", citation_format="invalid")
        errors = cfg.validate()
        assert any("citation" in e.lower() for e in errors)
