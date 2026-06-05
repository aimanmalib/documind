"""AnswerAgent — Generate answers with citations from retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseAgent
from .retriever import RetrievalResult


@dataclass
class Answer:
    """An answer with citations."""

    question: str
    answer_text: str
    citations: list[dict[str, str]]
    confidence: str = "medium"
    reasoning: str = ""


class AnswerAgent(BaseAgent):
    """Generate answers grounded in retrieved document chunks."""

    name = "answerer"

    def _build_system_prompt(self) -> str:
        return (
            "You are DocuMind's answer agent. You answer questions STRICTLY based on "
            "the provided document excerpts. If the answer is not in the provided context, "
            "say so clearly. Always cite your sources using [Source N] notation. "
            "Be precise, factual, and concise."
        )

    def _build_context(self, chunks: list[RetrievalResult]) -> str:
        """Build context string from retrieved chunks."""
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(
                f"[Source {i + 1}] (doc: {chunk.document_id}, chunk: {chunk.chunk_id})\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    async def answer(
        self,
        question: str,
        chunks: list[RetrievalResult],
        *,
        detail_level: str = "normal",
    ) -> Answer:
        """Generate an answer based on question and retrieved chunks."""
        context = self._build_context(chunks)

        level_instruction = {
            "brief": "Give a brief 1-2 sentence answer.",
            "normal": "Give a clear, complete answer in a paragraph.",
            "detailed": "Give a detailed answer with explanation and context.",
        }.get(detail_level, "Give a clear, complete answer.")

        prompt = (
            f"Based on the following document excerpts, answer this question:\n\n"
            f"Question: {question}\n\n"
            f"Document excerpts:\n{context}\n\n"
            f"Instructions: {level_instruction}\n"
            f"Include [Source N] citations in your answer. "
            f"End with a confidence level (high/medium/low)."
        )

        response = await self._call_mimo(prompt)
        content = response.content

        citations = []
        for i, chunk in enumerate(chunks):
            if f"[Source {i + 1}]" in content:
                citations.append(
                    {
                        "source_num": str(i + 1),
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "excerpt": chunk.text[:200],
                    }
                )

        confidence = "medium"
        lower = content.lower()
        if "confidence: high" in lower or "confidence level: high" in lower:
            confidence = "high"
        elif "confidence: low" in lower or "confidence level: low" in lower:
            confidence = "low"

        return Answer(
            question=question,
            answer_text=content,
            citations=citations,
            confidence=confidence,
            reasoning=response.reasoning_content,
        )

    async def run(self, **kwargs: Any) -> Answer:
        """Run answering. Expects 'question' and 'chunks' kwargs."""
        return await self.answer(
            kwargs["question"],
            kwargs["chunks"],
            detail_level=kwargs.get("detail_level", "normal"),
        )
