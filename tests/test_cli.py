"""Tests for CLI."""

from __future__ import annotations

import json

import httpx
from click.testing import CliRunner

from documind.cli import main
from documind.client import LLMClient


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

    def test_version(self):
        result = CliRunner().invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "documind" in result.output

    def test_ask_requires_question(self, sample_doc_path):
        result = CliRunner().invoke(main, ["ask", str(sample_doc_path)])
        assert result.exit_code == 1
        assert "-q" in result.output

    def test_ingest_runs_without_llm(self, monkeypatch, sample_doc_path):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-unused")
        result = CliRunner().invoke(main, ["ingest", str(sample_doc_path)])
        assert result.exit_code == 0, result.output
        assert "test_document.md" in result.output

    def test_summarize_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["summarize", "--help"])
        assert result.exit_code == 0


def _fake_llm(monkeypatch, replies: list[str]) -> list[dict]:
    """Route the real client through httpx.MockTransport; return captured bodies."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        text = replies[min(len(calls) - 1, len(replies) - 1)]
        return httpx.Response(
            200,
            json={
                "model": "fake",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    async def _get_client(self):
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return self._http

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(LLMClient, "_get_client", _get_client)
    return calls


class TestCLIEndToEnd:
    """Run real CLI commands end to end against a fake HTTP endpoint."""

    def test_ask_answers_and_exports(self, monkeypatch, sample_doc_path, tmp_path):
        calls = _fake_llm(monkeypatch, ["The pipeline uses agents [Source 1]."])
        out = tmp_path / "session.md"
        result = CliRunner().invoke(
            main,
            [
                "ask",
                str(sample_doc_path),
                "-q",
                "What does the pipeline use?",
                "--export",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "The pipeline uses agents" in result.output
        assert "Token usage" in result.output
        assert len(calls) == 1
        assert calls[0]["model"] == "gpt-4o-mini"
        assert "What does the pipeline use?" in calls[0]["messages"][-1]["content"]
        assert "The pipeline uses agents" in out.read_text()

    def test_ask_with_verify(self, monkeypatch, sample_doc_path):
        calls = _fake_llm(
            monkeypatch,
            ["Agents [Source 1].", "Claim 1: VERIFIED\nOVERALL: verified"],
        )
        result = CliRunner().invoke(
            main, ["ask", str(sample_doc_path), "-q", "What?", "--verify"]
        )
        assert result.exit_code == 0, result.output
        assert "Verification: verified" in result.output
        assert len(calls) == 2

    def test_summarize(self, monkeypatch, sample_doc_path):
        _fake_llm(monkeypatch, ["A short summary."])
        result = CliRunner().invoke(main, ["summarize", str(sample_doc_path)])
        assert result.exit_code == 0, result.output
        assert "A short summary." in result.output
