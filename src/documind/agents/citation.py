"""CitationAgent — Format citations in APA/MLA/Chicago styles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseAgent


@dataclass
class FormattedCitation:
    """A formatted citation."""

    style: str
    inline: str
    full_reference: str
    source_num: int


class CitationAgent(BaseAgent):
    """Generate properly formatted citations."""

    name = "citation"

    def _format_apa(
        self, title: str, author: str, year: str, source_num: int
    ) -> FormattedCitation:
        inline = f"({author}, {year})"
        full = f"{author} ({year}). {title}. Retrieved from document analysis."
        return FormattedCitation("apa", inline, full, source_num)

    def _format_mla(
        self, title: str, author: str, year: str, source_num: int
    ) -> FormattedCitation:
        inline = f"({author})"
        full = f'{author}. "{title}." {year}.'
        return FormattedCitation("mla", inline, full, source_num)

    def _format_chicago(
        self, title: str, author: str, year: str, source_num: int
    ) -> FormattedCitation:
        inline = f"({author} {year})"
        full = f'{author}. "{title}." {year}.'
        return FormattedCitation("chicago", inline, full, source_num)

    def format_citation(
        self,
        title: str,
        author: str = "Document",
        year: str = "n.d.",
        style: str | None = None,
        source_num: int = 1,
    ) -> FormattedCitation:
        """Format a single citation in the specified style."""
        fmt = style or self.config.citation_format
        formatters = {
            "apa": self._format_apa,
            "mla": self._format_mla,
            "chicago": self._format_chicago,
        }
        formatter = formatters.get(fmt, self._format_apa)
        return formatter(title, author, year, source_num)

    async def generate_references(
        self,
        citations: list[dict[str, str]],
        style: str | None = None,
    ) -> str:
        """Generate a formatted reference list using MiMo."""
        if not citations:
            return "No sources cited."

        source_list = "\n".join(
            f"Source {c.get('source_num', i + 1)}: chunk={c.get('chunk_id', '?')}, "
            f"doc={c.get('document_id', '?')}, excerpt={c.get('excerpt', '')[:100]}"
            for i, c in enumerate(citations)
        )

        fmt = style or self.config.citation_format
        prompt = (
            f"Format these sources as a {fmt.upper()} reference list:\n\n{source_list}\n\n"
            f"Use proper {fmt.upper()} formatting. Number each entry."
        )

        response = await self._call_mimo(prompt)
        return response.content

    async def run(self, **kwargs: Any) -> Any:
        """Run citation formatting."""
        if "citations" in kwargs:
            return await self.generate_references(
                kwargs["citations"], kwargs.get("style")
            )
        return self.format_citation(
            title=kwargs.get("title", "Untitled"),
            author=kwargs.get("author", "Document"),
            year=kwargs.get("year", "n.d."),
            style=kwargs.get("style"),
            source_num=kwargs.get("source_num", 1),
        )
