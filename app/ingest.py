"""PDF parsing and chunking."""
from dataclasses import dataclass
from pypdf import PdfReader
import io
import re


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    text: str


def parse_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Returns list of (page_number, text) for each page, 1-indexed."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def chunk_pages(
    doc_id: str,
    pages: list[tuple[int, str]],
    chunk_size: int = 250,
    overlap: int = 100,
) -> list[Chunk]:
    """Chunk each page's text into overlapping character windows.

    Chunking is per-page (never crosses a page boundary) so every chunk
    has one unambiguous page number for citation.
    """
    chunks: list[Chunk] = []
    counter = 0
    for page_num, text in pages:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        sentences = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > chunk_size:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)

        for piece in pieces:
            if piece:
                chunk_id = f"c-{counter:04d}"
                chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, page=page_num, text=piece))
                counter += 1
    return chunks
