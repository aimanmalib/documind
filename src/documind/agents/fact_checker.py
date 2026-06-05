"""FactCheckAgent — Verify answer claims against source documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseAgent
from .answerer import Answer
from .retriever import RetrievalResult


@dataclass
class FactCheckResult:
    """Result of fact-checking an answer."""

    overall_verdict: (
        str  # "verified", "partially_verified", "unverified", "contradicted"
    )
    claims_checked: int
    claims_verified: int
    claims_contradicted: int
    issues: list[str]
    details: str


class FactCheckAgent(BaseAgent):
    """Verify answer claims against source documents."""

    name = "fact_checker"

    def _build_system_prompt(self) -> str:
        return (
            "You are DocuMind's fact-checking agent. Compare claims in an answer against "
            "source documents. For each claim, determine if it is: VERIFIED (supported by source), "
            "UNVERIFIED (not found in source), or CONTRADICTED (conflicts with source). "
            "Be strict and precise."
        )

    async def check(
        self,
        answer: Answer,
        source_chunks: list[RetrievalResult],
    ) -> FactCheckResult:
        """Fact-check an answer against source documents."""
        sources_text = "\n\n".join(
            f"[Source {i + 1}]: {chunk.text}" for i, chunk in enumerate(source_chunks)
        )

        prompt = (
            f"Fact-check this answer against the source documents:\n\n"
            f"Answer: {answer.answer_text}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"List each factual claim, then rate it as VERIFIED, UNVERIFIED, or CONTRADICTED.\n"
            f"End with a summary line: OVERALL: [verified/partially_verified/unverified/contradicted]"
        )

        response = await self._call_mimo(prompt)
        content = response.content

        verified = (
            content.lower().count("verified")
            - content.lower().count("unverified")
            - content.lower().count("contradicted")
        )
        contradicted = content.lower().count("contradicted")
        unverified = content.lower().count("unverified")
        claims = max(verified + contradicted + unverified, 1)

        overall = "verified"
        if "contradicted" in content.lower():
            overall = "partially_verified" if verified > 0 else "contradicted"
        elif "unverified" in content.lower():
            overall = "partially_verified" if verified > 0 else "unverified"

        issues = []
        for line in content.split("\n"):
            lower = line.lower()
            if "contradicted" in lower or "unverified" in lower:
                issues.append(line.strip())

        return FactCheckResult(
            overall_verdict=overall,
            claims_checked=claims,
            claims_verified=max(verified, 0),
            claims_contradicted=contradicted,
            issues=issues[:10],
            details=content,
        )

    async def run(self, **kwargs: Any) -> FactCheckResult:
        """Run fact checking. Expects 'answer' and 'source_chunks'."""
        return await self.check(kwargs["answer"], kwargs["source_chunks"])
