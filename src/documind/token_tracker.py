"""Per-agent token usage tracking and reporting."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentUsage:
    """Token usage for a single agent call."""

    agent_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """Track and report token usage across all agents."""

    # Estimated daily token consumption per agent (millions)
    DAILY_ESTIMATES: dict[str, float] = {
        "ingester": 0.15,
        "indexer": 0.20,
        "retriever": 0.45,
        "answerer": 0.80,
        "summarizer": 0.35,
        "fact_checker": 0.25,
        "citation": 0.10,
        "export": 0.05,
        "pipeline": 0.05,
    }

    def __init__(self) -> None:
        self._records: list[AgentUsage] = []
        self._session_start: float = time.time()

    def record(
        self,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float = 0.0,
    ) -> AgentUsage:
        """Record a single agent API call."""
        usage = AgentUsage(
            agent_name=agent_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )
        self._records.append(usage)
        return usage

    def by_agent(self) -> dict[str, dict[str, Any]]:
        """Aggregate usage by agent name."""
        agg: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "calls": 0,
                "prompt": 0,
                "completion": 0,
                "total": 0,
                "latency_ms": 0.0,
            }
        )
        for r in self._records:
            agg[r.agent_name]["calls"] += 1
            agg[r.agent_name]["prompt"] += r.prompt_tokens
            agg[r.agent_name]["completion"] += r.completion_tokens
            agg[r.agent_name]["total"] += r.total_tokens
            agg[r.agent_name]["latency_ms"] += r.latency_ms
        return dict(agg)

    def total_tokens(self) -> int:
        """Total tokens consumed this session."""
        return sum(r.total_tokens for r in self._records)

    def total_calls(self) -> int:
        """Total API calls this session."""
        return len(self._records)

    def session_duration(self) -> float:
        """Seconds since tracker creation."""
        return time.time() - self._session_start

    def daily_projection(self) -> dict[str, float]:
        """Project daily token usage based on current rate."""
        duration = self.session_duration()
        if duration < 1:
            return {}
        tokens_per_sec = self.total_tokens() / duration
        projected_daily = tokens_per_sec * 86400
        return {
            "tokens_per_second": round(tokens_per_sec, 2),
            "projected_daily_tokens": round(projected_daily),
            "projected_daily_millions": round(projected_daily / 1_000_000, 2),
        }

    def summary(self) -> dict[str, Any]:
        """Full session summary."""
        return {
            "session_duration_sec": round(self.session_duration(), 1),
            "total_calls": self.total_calls(),
            "total_tokens": self.total_tokens(),
            "by_agent": self.by_agent(),
            "daily_estimates_millions": self.DAILY_ESTIMATES,
            "daily_total_estimated_millions": sum(self.DAILY_ESTIMATES.values()),
            "projection": self.daily_projection(),
        }

    def export_json(self, path: str | Path) -> Path:
        """Export summary to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.summary(), indent=2))
        return p

    def reset(self) -> None:
        """Clear all recorded usage."""
        self._records.clear()
        self._session_start = time.time()
