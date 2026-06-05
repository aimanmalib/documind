"""RetrieverAgent — Find relevant chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseAgent
from .indexer import IndexAgent


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    section: str = ""


class RetrieverAgent(BaseAgent):
    """Retrieve the most relevant document chunks for a query."""

    name = "retriever"

    def __init__(
        self, *args: Any, index: IndexAgent | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.index = index

    async def retrieve(
        self, query: str, *, top_k: int | None = None, min_score: float = 0.0
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks using keyword index + MiMo reranking."""
        if not self.index:
            raise ValueError("IndexAgent not set. Pass index= to constructor.")

        k = top_k or self.config.max_retrieval_k
        keyword_results = self.index.search_keywords(query, top_k=k * 2)

        if not keyword_results:
            return []

        results: list[RetrievalResult] = []
        for chunk_id, score in keyword_results:
            if score < min_score:
                continue
            entry = self.index.get_entry(chunk_id)
            if entry:
                results.append(
                    RetrievalResult(
                        chunk_id=entry.chunk_id,
                        document_id=entry.document_id,
                        text=entry.text,
                        score=round(score, 4),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    async def rerank_with_mimo(
        self, query: str, candidates: list[RetrievalResult], top_k: int = 3
    ) -> list[RetrievalResult]:
        """Use MiMo to rerank candidates by semantic relevance."""
        if not candidates:
            return []

        context_chunks = []
        for i, c in enumerate(candidates):
            context_chunks.append(f"[{i}] {c.text[:300]}")
        context = "\n\n".join(context_chunks)

        prompt = (
            f'Given the question: "{query}"\n\n'
            f"Rank these text chunks by relevance (most relevant first). "
            f"Return ONLY the indices in order, comma-separated.\n\n{context}"
        )

        response = await self._call_mimo(prompt)
        try:
            indices = [int(x.strip()) for x in response.content.strip().split(",")]
            ranked = [candidates[i] for i in indices if 0 <= i < len(candidates)]
            return ranked[:top_k]
        except (ValueError, IndexError):
            return candidates[:top_k]

    async def run(self, **kwargs: Any) -> list[RetrievalResult]:
        """Run retrieval. Expects 'query' kwarg."""
        query = kwargs["query"]
        results = await self.retrieve(query)
        if kwargs.get("rerank", False):
            results = await self.rerank_with_mimo(query, results)
        return results
