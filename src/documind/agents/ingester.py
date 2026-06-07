"""IngestAgent — Parse and chunk documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from html.parser import HTMLParser
from .base import BaseAgent


@dataclass
class TextChunk:
    """A chunk of text from a document."""

    chunk_id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    page_number: int | None = None
    section: str = ""
    word_count: int = 0
    char_count: int = 0

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


@dataclass
class Document:
    """Parsed document with metadata."""

    doc_id: str
    filename: str
    content: str
    file_type: str
    page_count: int = 0
    chunks: list[TextChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.chunks)


class IngestAgent(BaseAgent):
    """Parse documents and split into chunks."""

    name = "ingester"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chunk_size = self.config.max_chunk_size
        self.chunk_overlap = self.config.chunk_overlap

    def _detect_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        type_map = {
            ".txt": "text",
            ".md": "markdown",
            ".pdf": "pdf",
            ".docx": "docx",
            ".html": "html",
            ".htm": "html",
            ".json": "json",
            ".csv": "csv",
            ".rst": "rst",
        }
        return type_map.get(ext, "text")

    def _read_file(self, path: Path) -> str:
        """Read file content as text."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return self._read_docx(path)
        if suffix in (".html", ".htm"):
            return self._read_html(path)
        return path.read_text(encoding="utf-8", errors="replace")

    def _read_pdf(self, path: Path) -> str:
        """Extract text from PDF. Tries pymupdf, falls back to raw read."""
        try:
            import fitz

            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            return path.read_text(encoding="utf-8", errors="replace")

    def _read_docx(self, path: Path) -> str:
        """Extract text from DOCX. Tries python-docx, falls back."""
        try:
            from docx import Document as DocxDoc

            doc = DocxDoc(str(path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            return path.read_text(encoding="utf-8", errors="replace")

    def _read_html(self, path: Path) -> str:
        """Extract text from HTML while preserving headings"""
        class HeadingHTMLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts=[]
                self.ignore_tags={'script', 'style', 'head', 'meta', 'noscript'}
                self.ignore_depth=0

            def handle_starttag(self, tag, attrs):
                if tag in self.ignore_tags:
                    self.ignore_depth += 1
                elif tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
                    self.text_parts.append(f"\n\n{'#' * int(tag[1])} ")
                elif tag in {'p', 'div', 'br', 'li', 'article', 'section'}:
                    self.text_parts.append("\n")

            def handle_endtag(self, tag):
                if tag in self.ignore_tags:
                    self.ignore_depth = max(0, self.ignore_depth-1)
                elif tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li', 'article', 'section'}:
                    self.text_parts.append("\n\n")
            
            def handle_data(self, data):
                if self.ignore_depth == 0:
                    self.text_parts.append(re.sub(r'[ \t\r\f\v]+', ' ', data))
    
        raw = path.read_text(encoding="utf-8", errors="replace")
        parser = HeadingHTMLParser()
        parser.feed(raw)
        text = "".join(parser.text_parts)
        return re.sub(r'\n{3,}', '\n\n', text).strip()    
    


    def _generate_doc_id(self, content: str, filename: str) -> str:
        h = hashlib.sha256(content[:4096].encode()).hexdigest()[:12]
        return f"doc-{h}"

    def _split_into_chunks(self, text: str, doc_id: str) -> list[TextChunk]:
        """Split text into overlapping chunks."""
        chunks: list[TextChunk] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current = ""
        offset = 0
        chunk_idx = 0

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > self.chunk_size and current:
                cid = f"{doc_id}-chunk-{chunk_idx:04d}"
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        document_id=doc_id,
                        text=current.strip(),
                        start_offset=offset,
                        end_offset=offset + len(current),
                    )
                )
                overlap_text = (
                    current[-self.chunk_overlap :] if self.chunk_overlap else ""
                )
                offset += len(current) - len(overlap_text)
                current = overlap_text + " " + sentence
                chunk_idx += 1
            else:
                current = (current + " " + sentence).strip()

        if current.strip():
            cid = f"{doc_id}-chunk-{chunk_idx:04d}"
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    document_id=doc_id,
                    text=current.strip(),
                    start_offset=offset,
                    end_offset=offset + len(current),
                )
            )

        return chunks

    def _extract_sections(self, text: str) -> list[tuple[str, str]]:
        """Extract section headers and content from markdown/text."""
        sections: list[tuple[str, str]] = []
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(header_pattern.finditer(text))

        if not matches:
            return [("main", text)]

        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                sections.append((header, content))

        return sections

    async def run(
        self,
        path: str | Path,
        *,
        extract_sections: bool = True,
    ) -> Document:
        """Ingest a document: read, parse, chunk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Document not found: {p}")

        content = self._read_file(p)
        doc_id = self._generate_doc_id(content, p.name)
        file_type = self._detect_type(p)

        doc = Document(
            doc_id=doc_id,
            filename=p.name,
            content=content,
            file_type=file_type,
            metadata={"source_path": str(p.resolve()), "size_bytes": p.stat().st_size},
        )

        if extract_sections and file_type in ("markdown", "rst", "html"):
            sections = self._extract_sections(content)
            all_chunks: list[TextChunk] = []
            for section_name, section_text in sections:
                section_chunks = self._split_into_chunks(section_text, doc_id)
                for c in section_chunks:
                    c.section = section_name
                all_chunks.extend(section_chunks)
            doc.chunks = all_chunks
        else:
            doc.chunks = self._split_into_chunks(content, doc_id)

        return doc

    async def run_batch(self, paths: list[str | Path]) -> list[Document]:
        """Ingest multiple documents."""
        docs: list[Document] = []
        for p in paths:
            doc = await self.run(p)
            docs.append(doc)
        return docs
