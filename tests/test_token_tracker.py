"""Tests for token tracker."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from documind.token_tracker import AgentUsage, TokenTracker


class TestTokenTracker:
    """Test the TokenTracker class."""

    def test_initial_state(self):
        t = TokenTracker()
        assert t.total_tokens() == 0
        assert t.total_calls() == 0

    def test_record_single_call(self):
        t = TokenTracker()
        usage = t.record("answerer", 100, 50, 200.0)
        assert isinstance(usage, AgentUsage)
        assert usage.agent_name == "answerer"
        assert usage.total_tokens == 150

    def test_record_multiple_calls(self):
        t = TokenTracker()
        t.record("ingester", 200, 100, 150.0)
        t.record("indexer", 300, 150, 200.0)
        t.record("answerer", 400, 200, 300.0)
        assert t.total_calls() == 3
        assert t.total_tokens() == 1350

    def test_by_agent_aggregation(self):
        t = TokenTracker()
        t.record("answerer", 100, 50, 100.0)
        t.record("answerer", 200, 100, 200.0)
        t.record("ingester", 150, 75, 150.0)

        by_agent = t.by_agent()
        assert "answerer" in by_agent
        assert by_agent["answerer"]["calls"] == 2
        assert by_agent["answerer"]["total"] == 450
        assert by_agent["ingester"]["calls"] == 1

    def test_daily_estimates_populated(self):
        t = TokenTracker()
        assert "answerer" in t.DAILY_ESTIMATES
        assert "ingester" in t.DAILY_ESTIMATES
        assert sum(t.DAILY_ESTIMATES.values()) > 0

    def test_summary_structure(self):
        t = TokenTracker()
        t.record("test", 100, 50, 100.0)
        summary = t.summary()
        assert "total_calls" in summary
        assert "total_tokens" in summary
        assert "by_agent" in summary
        assert "daily_estimates_millions" in summary

    def test_export_json(self, tmp_path):
        t = TokenTracker()
        t.record("test", 100, 50, 100.0)
        out = t.export_json(tmp_path / "report.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_calls"] == 1

    def test_reset(self):
        t = TokenTracker()
        t.record("test", 100, 50, 100.0)
        assert t.total_calls() == 1
        t.reset()
        assert t.total_calls() == 0
        assert t.total_tokens() == 0

    def test_session_duration_positive(self):
        import time
        t = TokenTracker()
        time.sleep(0.01)
        assert t.session_duration() > 0
