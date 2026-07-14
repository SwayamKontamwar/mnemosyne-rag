from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .models import Chunk, SourceDocument

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


def discover(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in SUPPORTED:
            yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED:
            yield candidate


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> list[SourceDocument]:
    absolute = str(path.resolve())
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        documents: list[SourceDocument] = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(SourceDocument(absolute, path.stem, text, index + 1))
        return documents
    return [SourceDocument(absolute, path.stem, path.read_text(encoding="utf-8"))]


def chunk_document(doc: SourceDocument, size: int, overlap: int) -> list[Chunk]:
    lines = doc.text.splitlines()
    chunks: list[Chunk] = []
    cursor = 0
    ordinal = 0
    while cursor < len(lines):
        selected: list[str] = []
        chars = 0
        end = cursor
        while end < len(lines) and (chars < size or not selected):
            selected.append(lines[end])
            chars += len(lines[end]) + 1
            end += 1
        text = "\n".join(selected).strip()
        if text:
            chunks.append(
                Chunk(
                    document_path=doc.path,
                    title=doc.title,
                    text=text,
                    ordinal=ordinal,
                    start_line=None if doc.page else cursor + 1,
                    end_line=None if doc.page else end,
                    page=doc.page,
                )
            )
            ordinal += 1
        if end >= len(lines):
            break
        overlap_lines = 0
        overlap_chars = 0
        for line in reversed(selected):
            if overlap_chars >= overlap:
                break
            overlap_chars += len(line) + 1
            overlap_lines += 1
        cursor = max(cursor + 1, end - overlap_lines)
    return chunks
