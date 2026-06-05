"""OpenAI-compatible HTTP client for DocuMind.

Works with any provider that speaks the OpenAI ``/chat/completions`` protocol
(OpenAI, OpenRouter, Ollama, llama.cpp, Xiaomi MiMo Token Plan, ...). The auth
header style (bearer vs api-key) comes from the config's provider preset.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from .config import DocuMindConfig


@dataclass
class CompletionResponse:
    """Parsed response from a chat completion."""

    content: str = ""
    reasoning_content: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0


class LLMClient:
    """Async HTTP client for any OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: DocuMindConfig | None = None) -> None:
        self.config = config or DocuMindConfig.from_env()
        self._http: httpx.AsyncClient | None = None
        self._request_count: int = 0
        self._total_tokens: int = 0

    @property
    def base_url(self) -> str:
        return self.config.mimo_base_url.rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        """Build request headers (bearer or api-key, per provider preset)."""
        return self.config.headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        return self._http

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> CompletionResponse:
        """Send a chat completion request to MiMo API."""
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model or self.config.mimo_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        start = time.monotonic()
        if stream:
            return await self._stream_completion(client, payload, start)

        resp = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        data = await resp.json()
        elapsed = (time.monotonic() - start) * 1000

        self._request_count += 1
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        self._total_tokens += tokens

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return CompletionResponse(
            content=msg.get("content", ""),
            reasoning_content=msg.get("reasoning_content", ""),
            model=data.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=tokens,
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=round(elapsed, 2),
        )

    async def _stream_completion(
        self, client: httpx.AsyncClient, payload: dict[str, Any], start: float
    ) -> CompletionResponse:
        """Handle SSE streaming response."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        model = ""
        finish_reason = ""

        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                model = chunk.get("model", model)
                usage = chunk.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)

                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])
                    if "reasoning_content" in delta and delta["reasoning_content"]:
                        reasoning_parts.append(delta["reasoning_content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        elapsed = (time.monotonic() - start) * 1000
        total = prompt_tokens + completion_tokens
        self._request_count += 1
        self._total_tokens += total

        return CompletionResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            finish_reason=finish_reason,
            latency_ms=round(elapsed, 2),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    @property
    def stats(self) -> dict[str, Any]:
        """Return client usage statistics."""
        return {
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }


# Backward-compatible alias. New code should use LLMClient.
MiMoClient = LLMClient
