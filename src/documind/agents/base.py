"""Base agent abstract class for DocuMind pipeline."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..client import CompletionResponse, MiMoClient
from ..config import DocuMindConfig
from ..token_tracker import TokenTracker


class BaseAgent(ABC):
    """Abstract base class for all DocuMind agents."""

    name: str = "base"

    def __init__(
        self,
        client: MiMoClient | None = None,
        config: DocuMindConfig | None = None,
        tracker: TokenTracker | None = None,
    ) -> None:
        self.config = config or DocuMindConfig.from_env()
        self.client = client or MiMoClient(self.config)
        self.tracker = tracker or TokenTracker()

    def _build_system_prompt(self) -> str:
        """Build system prompt for this agent. Override in subclasses."""
        return (
            f"You are {self.name}, a specialized agent in the DocuMind "
            "document analysis pipeline. Be precise and factual."
        )

    async def _call_mimo(
        self, user_prompt: str, *, system_prompt: str | None = None, **kwargs: Any
    ) -> CompletionResponse:
        """Make a MiMo API call and track token usage."""
        messages: list[dict[str, str]] = []
        sys_prompt = system_prompt or self._build_system_prompt()
        messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})

        start = time.monotonic()
        response = await self.client.chat_completion(messages, **kwargs)
        elapsed = (time.monotonic() - start) * 1000

        self.tracker.record(
            agent_name=self.name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=elapsed,
        )
        return response

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any:
        """Execute the agent's primary task."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} agent={self.name!r}>"
