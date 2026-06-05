"""Tests for CLI."""

from __future__ import annotations

from click.testing import CliRunner

from documind.cli import main


class TestCLI:
    """Test the Click CLI."""

    def test_main_runs(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "DocuMind" in result.output

    def test_ingest_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["ingest", "--help"])
        assert result.exit_code == 0

    def test_ask_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["ask", "--help"])
        assert result.exit_code == 0
        assert "question" in result.output.lower()

    def test_stats_runs(self):
        runner = CliRunner()
        result = runner.invoke(main, ["stats"])
        assert result.exit_code == 0

    def test_summarize_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["summarize", "--help"])
        assert result.exit_code == 0
