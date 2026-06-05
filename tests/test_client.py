"""Tests for MiMo HTTP client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from documind.client import CompletionResponse, MiMoClient


def _make_mock_response(data: dict):
    """Create a mock httpx response with async json()."""
    resp = AsyncMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()

    # json() must be a coroutine since client uses `await resp.json()`
    async def _json():
        return data

    resp.json = _json
    return resp


class TestMiMoClient:
    """Test the MiMo HTTP client."""

    def test_init_with_config(self, config):
        client = MiMoClient(config)
        assert client.config.mimo_api_key == "test-key-123"

    def test_base_url_strips_trailing_slash(self, config):
        config.mimo_base_url = "https://example.com/v1/"
        client = MiMoClient(config)
        assert client.base_url == "https://example.com/v1"

    def test_headers_use_api_key_not_bearer(self, config):
        client = MiMoClient(config)
        headers = client.headers
        assert "api-key" in headers
        assert headers["api-key"] == "test-key-123"
        assert "Authorization" not in headers

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
