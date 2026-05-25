"""IndexAgent — Build keyword and TF-IDF index from chunks."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent
from .ingester import TextChunk


@dataclass
class IndexEntry:
    """An entry in the document index."""

    chunk_id: str
    document_id: str
    text: str
    keywords: list[str] = field(default_factory=list)
    tf_scores: dict[str, float] = field(default_factory=dict)


class IndexAgent(BaseAgent):
    """Build searchable index from document chunks."""

    name = "indexer"

    # Common English + Malay stop words
    STOP_WORDS: set[str] = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "about",
        "between", "through", "during", "before", "after", "above", "below",
        "and", "but", "or", "nor", "not", "so", "if", "then", "than", "that",
        "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
        "dan", "yang", "ini", "itu", "di", "ke", "dari", "untuk", "dengan",
        "pada", "adalah", "akan", "telah", "sudah", "belum", "tidak", "bukan",
        "juga", "atau", "tetapi", "kerana", "oleh", "seperti", "dalam",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._entries: dict[str, IndexEntry] = {}
        self._idf: dict[str, float] = {}
        self._doc_count: int = 0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words, removing stop words."""
        words = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
        return [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

    def _compute_tf(self, words: list[str]) -> dict[str, float]:
        """Compute term frequency for a word list."""
        counts = Counter(words)
        total = len(words) or 1
        return {w: c / total for w, c in counts.items()}

    def _compute_idf(self) -> None:
        """Compute inverse document frequency across all entries."""
        doc_freq: Counter[str] = Counter()
        for entry in self._entries.values():
            unique_terms = set(entry.tf_scores.keys())
            for term in unique_terms:
                doc_freq[term] += 1
        n = self._doc_count or 1
        self._idf = {
            term: math.log((n + 1) / (df + 1)) + 1 for term, df in doc_freq.items()
        }

    def _extract_keywords(self, tf: dict[str, float], top_n: int = 10) -> list[str]:
        """Extract top-N keywords by TF score."""
        sorted_terms = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        return [term for term, _ in sorted_terms[:top_n]]

    async def index_chunks(self, chunks: list[TextChunk]) -> int:
        """Index a list of text chunks. Returns count of indexed entries."""
        for chunk in chunks:
            words = self._tokenize(chunk.text)
            tf = self._compute_tf(words)
            keywords = self._extract_keywords(tf)

            self._entries[chunk.chunk_id] = IndexEntry(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                keywords=keywords,
                tf_scores=tf,
            )

        self._doc_count = len(self._entries)
        self._compute_idf()
        return len(chunks)

    def compute_tfidf(self, chunk_id: str) -> dict[str, float]:
        """Compute TF-IDF scores for a specific chunk."""
        entry = self._entries.get(chunk_id)
        if not entry:
            return {}
        return {
            term: tf * self._idf.get(term, 1.0)
            for term, tf in entry.tf_scores.items()
        }

    def search_keywords(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search index by keyword matching with TF-IDF scoring."""
        query_words = self._tokenize(query)
        if not query_words:
            return []

        scores: dict[str, float] = defaultdict(float)
        query_tf = self._compute_tf(query_words)

        for entry in self._entries.values():
            score = 0.0
            for word in query_words:
                if word in entry.tf_scores:
                    idf = self._idf.get(word, 1.0)
                    score += entry.tf_scores[word] * idf * query_tf.get(word, 0)
            if score > 0:
                scores[entry.chunk_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_entry(self, chunk_id: str) -> IndexEntry | None:
        """Retrieve a single index entry by chunk ID."""
        return self._entries.get(chunk_id)

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        total_keywords = sum(len(e.keywords) for e in self._entries.values())
        return {
            "total_entries": len(self._entries),
            "total_documents": self._doc_count,
            "unique_terms": len(self._idf),
            "avg_keywords_per_chunk": (
                total_keywords / len(self._entries) if self._entries else 0
            ),
        }

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the indexer. Expects 'chunks' kwarg."""
        chunks = kwargs.get("chunks", [])
        count = await self.index_chunks(chunks)
        return {"indexed": count, "stats": self.stats()}
