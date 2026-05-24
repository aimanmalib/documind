"""SummarizerAgent — Generate document-level summaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseAgent
from .ingester import Document


@dataclass
class Summary:
    """A document summary."""

    document_id: str
    filename: str
    summary_text: str
    key_points: list[str]
    word_count: int
    compression_ratio: float


class SummarizerAgent(BaseAgent):
    """Generate concise summaries of documents."""

    name = "summarizer"

    def _build_system_prompt(self) -> str:
        return (
            "You are DocuMind's summarization agent. Create clear, accurate summaries "
            "that capture the main ideas and key points. Use bullet points for key takeaways."
        )

    async def summarize(
        self,
        document: Document,
        *,
        max_length: str = "medium",
    ) -> Summary:
        """Summarize a full document."""
        content = document.content
        if len(content) > 12000:
            content = content[:12000] + "\n...[truncated]"

        length_instruction = {
            "short": "3-5 sentences",
            "medium": "1-2 paragraphs",
            "long": "3-4 paragraphs with detailed key points",
        }.get(max_length, "1-2 paragraphs")

        prompt = (
            f"Summarize the following document in {length_instruction}. "
            f"After the summary, list 3-7 key points as bullet points starting with '•'.\n\n"
            f"Document: {document.filename}\n{content}"
        )

        response = await self._call_mimo(prompt)
        text = response.content

        key_points = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("•") or stripped.startswith("- ") or stripped.startswith("* "):
                point = stripped.lstrip("•-* ").strip()
                if point:
                    key_points.append(point)

        words_original = document.total_words
        words_summary = len(text.split())
        ratio = words_summary / max(words_original, 1)

        return Summary(
            document_id=document.doc_id,
            filename=document.filename,
            summary_text=text,
            key_points=key_points,
            word_count=words_summary,
            compression_ratio=round(ratio, 3),
        )

    async def run(self, **kwargs: Any) -> Summary:
        """Run summarization. Expects 'document' kwarg."""
        return await self.summarize(
            kwargs["document"],
            max_length=kwargs.get("max_length", "medium"),
        )
