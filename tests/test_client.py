"""Tests for the OpenAI-compatible HTTP client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from documind.client import CompletionResponse, MiMoClient


def _make_mock_response(data: dict) -> httpx.Response:
    """Build a real httpx response (its json() is synchronous)."""
    request = httpx.Request("POST", "https://test.api.example.com/v1/chat/completions")
    return httpx.Response(200, json=data, request=request)


class TestMiMoClient:
    """Test the HTTP client."""

    def test_init_with_config(self, config):
        client = MiMoClient(config)
        assert client.config.mimo_api_key == "test-key-123"

    def test_base_url_strips_trailing_slash(self, config):
        config.mimo_base_url = "https://example.com/v1/"
        client = MiMoClient(config)
        assert client.base_url == "https://example.com/v1"

    def test_headers_use_bearer_by_default(self, config):
        client = MiMoClient(config)
        headers = client.headers
        assert headers["Authorization"] == "Bearer test-key-123"
        assert "api-key" not in headers

    def test_headers_content_type(self, config):
        client = MiMoClient(config)
        assert client.headers["Content-Type"] == "application/json"

    def test_initial_stats(self, config):
        client = MiMoClient(config)
        assert client.stats["request_count"] == 0
        assert client.stats["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_chat_completion_success(self, config):
        client = MiMoClient(config)
        mock_data = {
            "model": "mimo-v2.5-pro",
            "choices": [
                {
                    "message": {
                        "content": "Hello!",
                        "reasoning_content": "thinking...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        }

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_resp = _make_mock_response(mock_data)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_http.is_closed = False
            mock_get.return_value = mock_http

            result = await client.chat_completion([{"role": "user", "content": "Hi"}])
            assert isinstance(result, CompletionResponse)
            assert result.content == "Hello!"
            assert result.total_tokens == 60

    @pytest.mark.asyncio
    async def test_chat_completion_tracks_stats(self, config):
        client = MiMoClient(config)
        mock_data = {
            "model": "mimo-v2.5-pro",
            "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            },
        }
        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_resp = _make_mock_response(mock_data)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_http.is_closed = False
            mock_get.return_value = mock_http

            await client.chat_completion([{"role": "user", "content": "test"}])
            assert client.stats["request_count"] == 1
            assert client.stats["total_tokens"] == 300

    @pytest.mark.asyncio
    async def test_close(self, config):
        client = MiMoClient(config)
        mock_http = AsyncMock()
        mock_http.is_closed = False
        mock_http.aclose = AsyncMock()
        client._http = mock_http
        await client.close()
        mock_http.aclose.assert_called_once()


class TestCompletionResponse:
    """Test CompletionResponse dataclass."""

    def test_default_values(self):
        r = CompletionResponse()
        assert r.content == ""
        assert r.total_tokens == 0

    def test_custom_values(self):
        r = CompletionResponse(content="test", total_tokens=100)
        assert r.content == "test"
        assert r.total_tokens == 100


class TestRealHttpTransport:
    """Exercise the client through a real httpx stack (MockTransport)."""

    @pytest.mark.asyncio
    async def test_non_streaming_request_and_parse(self, config):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "model": "m",
                    "choices": [
                        {"message": {"content": "Paris"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 1,
                        "total_tokens": 8,
                    },
                },
            )

        client = MiMoClient(config)
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await client.chat_completion([{"role": "user", "content": "?"}])
        finally:
            await client.close()

        assert result.content == "Paris"
        assert result.total_tokens == 8
        assert seen["url"] == "https://test.api.example.com/v1/chat/completions"
        assert seen["auth"] == "Bearer test-key-123"
        assert seen["body"]["stream"] is False
        assert seen["body"]["model"] == config.model

    @pytest.mark.asyncio
    async def test_null_content_becomes_empty_string(self, config):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": None}}], "usage": {}},
            )

        client = MiMoClient(config)
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await client.chat_completion([{"role": "user", "content": "?"}])
        finally:
            await client.close()
        assert result.content == ""
